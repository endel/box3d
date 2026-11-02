// SPDX-FileCopyrightText: 2026 Erin Catto
// SPDX-License-Identifier: MIT

#pragma once

#include "core.h"

#include "box3d/types.h"

#include <wasm_simd128.h>

// wide float holds 4 numbers
typedef v128_t b3FloatW4;

static inline b3FloatW4 b3ZeroW4( void ) { return wasm_f32x4_splat( 0.0f ); }
static inline b3FloatW4 b3SplatW4( float scalar ) { return wasm_f32x4_splat( scalar ); }
static inline b3FloatW4 b3SetW4( float a, float b, float c, float d ) { return wasm_f32x4_make( a, b, c, d ); }
static inline b3FloatW4 b3LoadW4( const float* data ) { return wasm_v128_load( data ); }
static inline void b3StoreW4( float* data, b3FloatW4 a ) { wasm_v128_store( data, a ); }
static inline b3FloatW4 b3NegW4( b3FloatW4 a ) { return wasm_f32x4_neg( a ); }
static inline b3FloatW4 b3AbsW4( b3FloatW4 a ) { return wasm_f32x4_abs( a ); }
static inline b3FloatW4 b3AddW4( b3FloatW4 a, b3FloatW4 b ) { return wasm_f32x4_add( a, b ); }
static inline b3FloatW4 b3SubW4( b3FloatW4 a, b3FloatW4 b ) { return wasm_f32x4_sub( a, b ); }
static inline b3FloatW4 b3MulW4( b3FloatW4 a, b3FloatW4 b ) { return wasm_f32x4_mul( a, b ); }
static inline b3FloatW4 b3DivW4( b3FloatW4 a, b3FloatW4 b ) { return wasm_f32x4_div( a, b ); }
static inline b3FloatW4 b3SqrtW4( b3FloatW4 a ) { return wasm_f32x4_sqrt( a ); }

// Cannot use real FMA because it doesn't match the non-SIMD path
static inline b3FloatW4 b3MulAddW4( b3FloatW4 a, b3FloatW4 b, b3FloatW4 c ) { return wasm_f32x4_add( a, wasm_f32x4_mul( b, c ) ); }

// pmin and pmax read as SSE's min and max: a < b ? a : b. The IEEE ones cost more on x86.
static inline b3FloatW4 b3MinW4( b3FloatW4 a, b3FloatW4 b ) { return wasm_f32x4_pmin( b, a ); }
static inline b3FloatW4 b3MaxW4( b3FloatW4 a, b3FloatW4 b ) { return wasm_f32x4_pmax( b, a ); }

static inline b3FloatW4 b3AndW4( b3FloatW4 a, b3FloatW4 b ) { return wasm_v128_and( a, b ); }
static inline b3FloatW4 b3OrW4( b3FloatW4 a, b3FloatW4 b ) { return wasm_v128_or( a, b ); }

// a & ~b
static inline b3FloatW4 b3AndNotW4( b3FloatW4 a, b3FloatW4 b ) { return wasm_v128_andnot( a, b ); }

static inline b3FloatW4 b3SoftMaskW4( const int* indexA, const int* indexB )
{
	v128_t zero = wasm_i32x4_splat( 0 );
	return wasm_v128_or( wasm_i32x4_eq( wasm_v128_load( indexA ), zero ), wasm_i32x4_eq( wasm_v128_load( indexB ), zero ) );
}

static inline b3FloatW4 b3GreaterThanW4( b3FloatW4 a, b3FloatW4 b ) { return wasm_f32x4_gt( a, b ); }
static inline b3FloatW4 b3LessThanW4( b3FloatW4 a, b3FloatW4 b ) { return wasm_f32x4_lt( a, b ); }
static inline b3FloatW4 b3EqualsW4( b3FloatW4 a, b3FloatW4 b ) { return wasm_f32x4_eq( a, b ); }
static inline bool b3AllZeroW4( b3FloatW4 a ) { return wasm_i32x4_all_true( wasm_f32x4_eq( a, wasm_f32x4_splat( 0.0f ) ) ); }
static inline bool b3AnyTrueW4( b3FloatW4 mask ) { return wasm_v128_any_true( mask ); }
static inline int b3MoveMaskW4( b3FloatW4 mask ) { return (int)wasm_i32x4_bitmask( mask ); }

// component-wise returns mask ? b : a
static inline b3FloatW4 b3BlendW4( b3FloatW4 a, b3FloatW4 b, b3FloatW4 mask ) { return wasm_v128_bitselect( b, a, mask ); }

static inline b3FloatW4 b3EmbedIndexW4( b3FloatW4 value, int baseIndex, int bitCount )
{
	uint32_t mask = ( 1u << bitCount ) - 1;
	v128_t index = wasm_i32x4_make( baseIndex, baseIndex + 1, baseIndex + 2, baseIndex + 3 );
	return wasm_v128_or( wasm_v128_and( value, wasm_u32x4_splat( ~mask ) ), index );
}

static inline int b3MinIndexW4( b3FloatW4 a, int bitCount )
{
	a = wasm_f32x4_pmin( wasm_i32x4_shuffle( a, a, 2, 3, 0, 1 ), a );
	a = wasm_f32x4_pmin( wasm_i32x4_shuffle( a, a, 1, 0, 3, 2 ), a );
	return (int)( (uint32_t)wasm_i32x4_extract_lane( a, 0 ) & ( ( 1u << bitCount ) - 1 ) );
}

typedef struct b3AABBV
{
	v128_t lower;
	v128_t upper;
} b3AABBV;

B3_FORCE_INLINE b3AABBV b3LoadAABBV( const b3AABB* aabb )
{
	const float* base = &aabb->lowerBound.x;

	// Offset to avoid reading off the end (avoid UB).
	// [lz ux uy uz]
	v128_t v1 = wasm_v128_load( base + 2 );
	b3AABBV result;
	// [lx ly lz -]
	result.lower = wasm_v128_load( base );
	// [ux uy uz -]
	result.upper = wasm_i32x4_shuffle( v1, v1, 1, 2, 3, 0 );
	return result;
}

B3_FORCE_INLINE bool b3OverlapAABBV( b3AABBV a, b3AABBV b )
{
	v128_t test = wasm_v128_and( wasm_f32x4_le( a.lower, b.upper ), wasm_f32x4_le( b.lower, a.upper ) );
	return ( wasm_i32x4_bitmask( test ) & 7 ) == 7;
}

B3_FORCE_INLINE bool b3OverlapNode( b3AABBV av, const b3TreeNode* node ) { return b3OverlapAABBV( av, b3LoadAABBV( &node->aabb ) ); }
B3_FORCE_INLINE bool b3OverlapV( const b3AABB* a, const b3AABB* b ) { return b3OverlapAABBV( b3LoadAABBV( a ), b3LoadAABBV( b ) ); }

B3_FORCE_INLINE b3AABBV b3UnionAABBV( b3AABBV a, b3AABBV b )
{
	b3AABBV result;
	result.lower = b3MinW4( a.lower, b.lower );
	result.upper = b3MaxW4( a.upper, b.upper );
	return result;
}

B3_FORCE_INLINE b3AABBV b3UnionPairV( const b3TreeNode* pair ) { return b3UnionAABBV( b3LoadAABBV( &pair[0].aabb ), b3LoadAABBV( &pair[1].aabb ) ); }

B3_FORCE_INLINE void b3StoreAABBV( b3AABB* aabb, b3AABBV value, bool condition )
{
	// [lx ly lz ux]
	v128_t raw0 = wasm_i32x4_shuffle( value.lower, value.upper, 0, 1, 2, 4 );
	// [lz ux uy uz]
	v128_t raw1 = wasm_i32x4_shuffle( raw0, value.upper, 2, 3, 5, 6 );

	float* base = &aabb->lowerBound.x;
	v128_t mask = wasm_i32x4_splat( condition ? -1 : 0 );
	wasm_v128_store( base, wasm_v128_bitselect( raw0, wasm_v128_load( base ), mask ) );
	wasm_v128_store( base + 2, wasm_v128_bitselect( raw1, wasm_v128_load( base + 2 ), mask ) );
}

B3_FORCE_INLINE void b3TransposeW4( b3FloatW4 r0, b3FloatW4 r1, b3FloatW4 r2, b3FloatW4 r3, b3FloatW4* c0, b3FloatW4* c1,
									b3FloatW4* c2, b3FloatW4* c3 )
{
	b3FloatW4 t0 = wasm_i32x4_shuffle( r0, r2, 0, 4, 1, 5 );
	b3FloatW4 t1 = wasm_i32x4_shuffle( r1, r3, 0, 4, 1, 5 );
	b3FloatW4 t2 = wasm_i32x4_shuffle( r0, r2, 2, 6, 3, 7 );
	b3FloatW4 t3 = wasm_i32x4_shuffle( r1, r3, 2, 6, 3, 7 );
	*c0 = wasm_i32x4_shuffle( t0, t1, 0, 4, 1, 5 );
	*c1 = wasm_i32x4_shuffle( t0, t1, 2, 6, 3, 7 );
	*c2 = wasm_i32x4_shuffle( t2, t3, 0, 4, 1, 5 );
	*c3 = wasm_i32x4_shuffle( t2, t3, 2, 6, 3, 7 );
}
