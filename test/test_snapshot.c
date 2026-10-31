// SPDX-FileCopyrightText: 2026 Endel Dreyer
// SPDX-License-Identifier: MIT

#include "test_macros.h"

#include "box3d/box3d.h"
#include "box3d/collision.h"

#include <stdlib.h>
#include <string.h>

#define COLUMN_COUNT 6
#define HEIGHT_COUNT 5

static void Step( b3WorldId worldId, int stepCount )
{
	for ( int i = 0; i < stepCount; ++i )
	{
		b3World_Step( worldId, 1.0f / 60.0f, 4 );
	}
}

// Boxes dropped on a mesh, some still falling: a save catches live contacts and warm starting.
static b3WorldId CreatePile( const b3MeshData* floor, b3BodyId* lastBodyId )
{
	b3WorldDef worldDef = b3DefaultWorldDef();
	b3WorldId worldId = b3CreateWorld( &worldDef );

	b3BodyDef groundDef = b3DefaultBodyDef();
	b3BodyId groundId = b3CreateBody( worldId, &groundDef );
	b3ShapeDef groundShapeDef = b3DefaultShapeDef();
	b3CreateMeshShape( groundId, &groundShapeDef, floor, b3Vec3_one );

	b3BoxHull box = b3MakeBoxHull( 0.5f, 0.5f, 0.5f );
	b3ShapeDef shapeDef = b3DefaultShapeDef();
	shapeDef.enableContactEvents = true;
	shapeDef.enableHitEvents = true;
	shapeDef.enableSensorEvents = true;
	b3BodyDef bodyDef = b3DefaultBodyDef();
	bodyDef.type = b3_dynamicBody;
	for ( int i = 0; i < COLUMN_COUNT * HEIGHT_COUNT; ++i )
	{
		bodyDef.position = (b3Pos){ 1.2f * ( i % COLUMN_COUNT ) + 0.1f * ( i / COLUMN_COUNT ), 1.0f + 1.1f * ( i / COLUMN_COUNT ), 0.0f };
		*lastBodyId = b3CreateBody( worldId, &bodyDef );
		b3CreateHullShape( *lastBodyId, &shapeDef, &box.base );
	}
	return worldId;
}

static b3WorldId CreateEmptyWorld( void )
{
	b3WorldDef worldDef = b3DefaultWorldDef();
	return b3CreateWorld( &worldDef );
}

// Two worlds at the same state write the same bytes, so comparing images covers everything saved.
static bool SameImage( b3WorldId worldA, b3SnapshotContext* contextA, b3WorldId worldB, b3SnapshotContext* contextB )
{
	const void* dataA;
	const void* dataB;
	int sizeA = b3World_SaveSnapshot( worldA, contextA, &dataA );
	int sizeB = b3World_SaveSnapshot( worldB, contextB, &dataB );
	return sizeA > 0 && sizeA == sizeB && memcmp( dataA, dataB, sizeA ) == 0;
}

// A loaded world goes on exactly as the one saved, from another world slot and an empty context.
static int ResumesExactly( void )
{
	b3MeshData* floor = b3CreateGridMesh( 20, 20, 2.0f, 1, true );
	b3BodyId bodyId;
	b3WorldId savedId = CreatePile( floor, &bodyId );
	b3SnapshotContext* savedContext = b3CreateSnapshotContext();
	Step( savedId, 30 );
	ENSURE( b3World_GetAwakeBodyCount( savedId ) > 0 );

	const void* data;
	int size = b3World_SaveSnapshot( savedId, savedContext, &data );
	ENSURE( size > 0 );

	b3WorldId loadedId = CreateEmptyWorld();
	b3SnapshotContext* loadedContext = b3CreateSnapshotContext();
	ENSURE( b3World_LoadSnapshot( loadedId, loadedContext, data, size ) );
	ENSURE( SameImage( savedId, savedContext, loadedId, loadedContext ) );

	b3BodyId loadedBodyId = bodyId;
	loadedBodyId.world0 = (uint16_t)( loadedId.index1 - 1 );
	ENSURE( b3Body_IsValid( loadedBodyId ) );

	b3Pos before = b3Body_GetPosition( bodyId );
	for ( int i = 0; i < 12; ++i )
	{
		Step( savedId, 10 );
		Step( loadedId, 10 );
		ENSURE( SameImage( savedId, savedContext, loadedId, loadedContext ) );
	}

	b3Pos after = b3Body_GetPosition( loadedBodyId );
	ENSURE( after.y != before.y );
	ENSURE( after.y == b3Body_GetPosition( bodyId ).y );

	b3DestroyWorld( savedId );
	b3DestroyWorld( loadedId );
	b3DestroySnapshotContext( savedContext );
	b3DestroySnapshotContext( loadedContext );
	b3DestroyMesh( floor );
	return 0;
}

// Geometry both contexts hold stays out of the image, and the load finds it by content.
static int SharedGeometryStaysOut( void )
{
	b3MeshData* floor = b3CreateGridMesh( 20, 20, 2.0f, 1, true );
	b3BodyId bodyId;
	b3WorldId savedId = CreatePile( floor, &bodyId );
	Step( savedId, 30 );

	b3SnapshotContext* plainContext = b3CreateSnapshotContext();
	const void* data;
	int plainSize = b3World_SaveSnapshot( savedId, plainContext, &data );

	b3SnapshotContext* savedContext = b3CreateSnapshotContext();
	b3SnapshotContext_AddWorld( savedContext, savedId );
	int size = b3World_SaveSnapshot( savedId, savedContext, &data );
	ENSURE( 0 < size && size + floor->byteCount / 2 < plainSize );

	// A peer's own setup, with its own copy of the mesh
	b3MeshData* peerFloor = b3CreateGridMesh( 20, 20, 2.0f, 1, true );
	b3WorldId loadedId = CreatePile( peerFloor, &bodyId );
	b3SnapshotContext* loadedContext = b3CreateSnapshotContext();
	b3SnapshotContext_AddWorld( loadedContext, loadedId );
	ENSURE( b3World_LoadSnapshot( loadedId, loadedContext, data, size ) );

	Step( savedId, 60 );
	Step( loadedId, 60 );
	ENSURE( SameImage( savedId, savedContext, loadedId, loadedContext ) );

	b3DestroyWorld( savedId );
	b3DestroyWorld( loadedId );
	b3DestroySnapshotContext( plainContext );
	b3DestroySnapshotContext( savedContext );
	b3DestroySnapshotContext( loadedContext );
	b3DestroyMesh( floor );
	b3DestroyMesh( peerFloor );
	return 0;
}

// Geometry neither the image nor the context holds refuses the load, before touching the world.
static int MissingGeometryRefused( void )
{
	b3MeshData* floor = b3CreateGridMesh( 20, 20, 2.0f, 1, true );
	b3BodyId bodyId;
	b3WorldId savedId = CreatePile( floor, &bodyId );
	b3SnapshotContext* savedContext = b3CreateSnapshotContext();
	b3SnapshotContext_AddWorld( savedContext, savedId );
	Step( savedId, 30 );

	const void* data;
	int size = b3World_SaveSnapshot( savedId, savedContext, &data );

	b3BodyId untouchedId;
	b3WorldId untouchedWorldId = CreatePile( floor, &untouchedId );
	b3SnapshotContext* emptyContext = b3CreateSnapshotContext();
	b3Pos before = b3Body_GetPosition( untouchedId );
	ENSURE( b3World_LoadSnapshot( untouchedWorldId, emptyContext, data, size ) == false );
	ENSURE( b3Body_GetPosition( untouchedId ).y == before.y );
	Step( untouchedWorldId, 10 );
	ENSURE( b3Body_GetPosition( untouchedId ).y != before.y );

	b3DestroyWorld( savedId );
	b3DestroyWorld( untouchedWorldId );
	b3DestroySnapshotContext( savedContext );
	b3DestroySnapshotContext( emptyContext );
	b3DestroyMesh( floor );
	return 0;
}

// An image cut anywhere is refused before touching the world.
static int TruncatedImageRefused( void )
{
	b3MeshData* floor = b3CreateGridMesh( 20, 20, 2.0f, 1, true );
	b3BodyId bodyId;
	b3WorldId savedId = CreatePile( floor, &bodyId );
	b3SnapshotContext* savedContext = b3CreateSnapshotContext();
	Step( savedId, 30 );

	const void* data;
	int size = b3World_SaveSnapshot( savedId, savedContext, &data );

	b3BodyId untouchedId;
	b3WorldId untouchedWorldId = CreatePile( floor, &untouchedId );
	b3SnapshotContext* context = b3CreateSnapshotContext();
	b3Pos before = b3Body_GetPosition( untouchedId );
	for ( int part = 0; part < 64; ++part )
	{
		ENSURE( b3World_LoadSnapshot( untouchedWorldId, context, data, size / 64 * part ) == false );
	}
	ENSURE( b3World_LoadSnapshot( untouchedWorldId, context, data, size - 1 ) == false );
	ENSURE( b3Body_GetPosition( untouchedId ).y == before.y );
	Step( untouchedWorldId, 10 );
	ENSURE( b3Body_GetPosition( untouchedId ).y != before.y );

	b3DestroyWorld( savedId );
	b3DestroyWorld( untouchedWorldId );
	b3DestroySnapshotContext( savedContext );
	b3DestroySnapshotContext( context );
	b3DestroyMesh( floor );
	return 0;
}

// What the last step reported reads the same after a load: it's there until the next step.
static int EventsTravel( void )
{
	b3MeshData* floor = b3CreateGridMesh( 20, 20, 2.0f, 1, true );
	b3BodyId bodyId;
	b3WorldId savedId = CreatePile( floor, &bodyId );

	b3BodyDef sensorBody = b3DefaultBodyDef();
	sensorBody.position = (b3Pos){ 3.0f, 2.0f, 0.0f };
	b3BodyId sensorId = b3CreateBody( savedId, &sensorBody );
	b3ShapeDef sensorDef = b3DefaultShapeDef();
	sensorDef.isSensor = true;
	sensorDef.enableSensorEvents = true;
	b3BoxHull zone = b3MakeBoxHull( 6.0f, 3.0f, 3.0f );
	b3CreateHullShape( sensorId, &sensorDef, &zone.base );

	b3SnapshotContext* savedContext = b3CreateSnapshotContext();
	int tries = 0;
	b3ContactEvents events = b3World_GetContactEvents( savedId );
	while ( events.beginCount == 0 && tries < 120 )
	{
		Step( savedId, 1 );
		events = b3World_GetContactEvents( savedId );
		tries += 1;
	}
	ENSURE( events.beginCount > 0 && events.hitCount > 0 );

	const void* data;
	int size = b3World_SaveSnapshot( savedId, savedContext, &data );
	b3WorldId loadedId = CreateEmptyWorld();
	b3SnapshotContext* loadedContext = b3CreateSnapshotContext();
	ENSURE( b3SnapshotContext_AddImage( loadedContext, data, size ) );
	ENSURE( b3SnapshotContext_AddImage( loadedContext, data, size / 2 ) == false );
	ENSURE( b3World_LoadSnapshot( loadedId, loadedContext, data, size ) );

	b3ContactEvents loaded = b3World_GetContactEvents( loadedId );
	ENSURE( loaded.beginCount == events.beginCount && loaded.endCount == events.endCount && loaded.hitCount == events.hitCount );
	ENSURE( loaded.beginEvents[0].shapeIdA.world0 == loadedId.index1 - 1 );
	ENSURE( loaded.beginEvents[0].shapeIdA.index1 == events.beginEvents[0].shapeIdA.index1 );
	ENSURE( b3Shape_IsValid( loaded.beginEvents[0].shapeIdA ) );
	b3SensorEvents saw = b3World_GetSensorEvents( savedId );
	b3SensorEvents sawLoaded = b3World_GetSensorEvents( loadedId );
	ENSURE( saw.beginCount == sawLoaded.beginCount && saw.endCount == sawLoaded.endCount );

	Step( savedId, 30 );
	Step( loadedId, 30 );
	ENSURE( SameImage( savedId, savedContext, loadedId, loadedContext ) );

	b3DestroyWorld( savedId );
	b3DestroyWorld( loadedId );
	b3DestroySnapshotContext( savedContext );
	b3DestroySnapshotContext( loadedContext );
	b3DestroyMesh( floor );
	return 0;
}

// A world that went its own way takes on the saved one, what it held replaced.
static int LoadsOverLiveWorld( void )
{
	b3MeshData* floor = b3CreateGridMesh( 20, 20, 2.0f, 1, true );
	b3BodyId bodyId;
	b3WorldId savedId = CreatePile( floor, &bodyId );
	b3SnapshotContext* savedContext = b3CreateSnapshotContext();
	b3SnapshotContext_AddWorld( savedContext, savedId );
	Step( savedId, 30 );

	b3BodyId liveBodyId;
	b3WorldId liveId = CreatePile( floor, &liveBodyId );
	b3SnapshotContext* liveContext = b3CreateSnapshotContext();
	b3SnapshotContext_AddWorld( liveContext, liveId );
	b3Body_ApplyLinearImpulseToCenter( liveBodyId, (b3Vec3){ 40.0f, 0.0f, 0.0f }, true );
	Step( liveId, 200 );

	for ( int i = 0; i < 3; ++i )
	{
		const void* data;
		int size = b3World_SaveSnapshot( savedId, savedContext, &data );
		ENSURE( b3World_LoadSnapshot( liveId, liveContext, data, size ) );
		Step( savedId, 20 );
		Step( liveId, 20 );
		ENSURE( SameImage( savedId, savedContext, liveId, liveContext ) );
	}

	b3DestroyWorld( savedId );
	b3DestroyWorld( liveId );
	b3DestroySnapshotContext( savedContext );
	b3DestroySnapshotContext( liveContext );
	b3DestroyMesh( floor );
	return 0;
}

int SnapshotTest( void )
{
	RUN_SUBTEST( ResumesExactly );
	RUN_SUBTEST( SharedGeometryStaysOut );
	RUN_SUBTEST( MissingGeometryRefused );
	RUN_SUBTEST( TruncatedImageRefused );
	RUN_SUBTEST( EventsTravel );
	RUN_SUBTEST( LoadsOverLiveWorld );
	return 0;
}
