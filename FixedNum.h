
#ifndef FIXED_NUM_H
#define FIXED_NUM_H

//#define TOTAL_BITS 32
//#define SIGNED
//#define FIXED_FBITS 14
#define MASK ((1 << FIXED_FBITS)-1)
#define ONE_FXP (1 << FIXED_FBITS)

//    [1, 2, 3],             # 8 bits
//    [1, 2, 4, 6],          # 16 bits
//    [1, 2, 4, 8, 10, 14]   # 32 bits

#ifdef SIGNED
#if TOTAL_BITS==8
typedef int8_t TYPE;
typedef int16_t TYPE_DOUBLE_SIZE;
#define INF_POS (TYPE)0x7F
#define INF_NEG (TYPE)0x80
#elif TOTAL_BITS==16
typedef int16_t TYPE; // 16 bits
typedef int32_t TYPE_DOUBLE_SIZE;
#define INF_POS (TYPE)0x7FFF
#define INF_NEG (TYPE)0x8000
#define POW_CF1 (TYPE)(5616 >> (14 - FIXED_FBITS))
#define POW_CF2 (TYPE)(10640 >> (14 - FIXED_FBITS))
#define POW_CF3 (TYPE)(16448 >> (14 - FIXED_FBITS))
#define EXP_CF (TYPE)(23632 >> (14 - FIXED_FBITS))
#define SQRT_CF1 (TYPE)((int16_t)(-577) >> (14 - FIXED_FBITS))
#define SQRT_CF2 (TYPE)(8229 >> (14 - FIXED_FBITS))
#define SQRT_CF3 (TYPE)(8957 >> (14 - FIXED_FBITS))
#else
typedef int32_t TYPE; // 32 bits
typedef int64_t TYPE_DOUBLE_SIZE;
#define INF_POS (TYPE)0x7FFFFFFF
#define INF_NEG (TYPE)0x80000000
#define POW_CF1 (TYPE)(92012544 >> (28 - FIXED_FBITS))
#define POW_CF2 (TYPE)(174325760 >> (28 - FIXED_FBITS))
#define POW_CF3 (TYPE)(269484032 >> (28 - FIXED_FBITS))
#define EXP_CF (TYPE)(387186688 >> (28 - FIXED_FBITS))
#define SQRT_CF1 (TYPE)((int32_t)(-9468632) >> (28 - FIXED_FBITS))
#define SQRT_CF2 (TYPE)(134833239 >> (28 - FIXED_FBITS))
#define SQRT_CF3 (TYPE)(146763591 >> (28 - FIXED_FBITS))
#endif

#else
#if TOTAL_BITS==8
typedef uint8_t TYPE;
typedef uint16_t TYPE_DOUBLE_SIZE;
#define INF_POS (TYPE)0xFF
#define INF_NEG (TYPE)0x00
#elif TOTAL_BITS==16
typedef uint16_t TYPE; // 16 bits
typedef uint32_t TYPE_DOUBLE_SIZE;
#define INF_POS (TYPE)0xFFFF
#define INF_NEG (TYPE)0x0000
#define POW_CF1 (TYPE)(5616 >> (14 - FIXED_FBITS))
#define POW_CF2 (TYPE)(10640 >> (14 - FIXED_FBITS))
#define POW_CF3 (TYPE)(16448 >> (14 - FIXED_FBITS))
#define EXP_CF (TYPE)(23632 >> (14 - FIXED_FBITS))
#define SQRT_CF1 (TYPE)((int16_t)(-577) >> (14 - FIXED_FBITS))
#define SQRT_CF2 (TYPE)(8229 >> (14 - FIXED_FBITS))
#define SQRT_CF3 (TYPE)(8957 >> (14 - FIXED_FBITS))
#else
typedef uint32_t TYPE; // 32 bits
typedef uint64_t TYPE_DOUBLE_SIZE;
#define INF_POS (TYPE)0xFFFFFFFF
#define INF_NEG (TYPE)0x00000000
#define POW_CF1 (TYPE)(92012544 >> (28 - FIXED_FBITS))
#define POW_CF2 (TYPE)(174325760 >> (28 - FIXED_FBITS))
#define POW_CF3 (TYPE)(269484032 >> (28 - FIXED_FBITS))
#define EXP_CF (TYPE)(387186688 >> (28 - FIXED_FBITS))
#define SQRT_CF1 (TYPE)((int32_t)(-9468632) >> (28 - FIXED_FBITS))
#define SQRT_CF2 (TYPE)(134833239 >> (28 - FIXED_FBITS))
#define SQRT_CF3 (TYPE)(146763591 >> (28 - FIXED_FBITS))
#endif
#endif

#define abs(x) ((x) < 0 ? (-(x)) : (x))

typedef TYPE FixedNum;

float getValue(FixedNum x){
	return (x / (float)((TYPE)1 << FIXED_FBITS));
}

FixedNum setValue(float x){
	(*(uint32_t *)&x) = (*(uint32_t *)&x) + ((uint32_t)FIXED_FBITS << 23);
  return (TYPE)round(x);
}

// ARITHMETIC OPERATIONS

FixedNum fxp_sum(const FixedNum left, const FixedNum right){
	#ifdef OVERFLOW_DETECT
	if (left == INF_POS || right == INF_POS)
		return INF_POS;
	if (left == INF_NEG || right == INF_NEG)
		return INF_NEG;
	#endif

	return (left + right);
}

FixedNum fxp_diff(const FixedNum left, const FixedNum right){
	#ifdef OVERFLOW_DETECT
	if (left == INF_POS || right == INF_NEG)
		return INF_POS;
	if (left == INF_NEG || right == INF_POS)
		return INF_NEG;
	#endif

	return (left - right);
}

FixedNum fxp_mul(const FixedNum left, const FixedNum right){
	if (left == 0 || right == 0)
		return ((TYPE)0);

	TYPE_DOUBLE_SIZE aux = (((TYPE_DOUBLE_SIZE)left * (TYPE_DOUBLE_SIZE)right) >> FIXED_FBITS);

	#ifdef OVERFLOW_DETECT
	if (aux > INF_POS)
		return (INF_POS);
	if (aux < INF_NEG)
		return (INF_NEG);
	#endif

	return ((TYPE)aux);
}

FixedNum fxp_div(const FixedNum left, const FixedNum right){
	#ifdef OVERFLOW_DETECT
	if (right == 0)
		return (left > 0 ? (INF_POS) : (INF_NEG));
  if (right == INF_POS || right == INF_NEG)
    return ((TYPE)0);
	#endif

	return ((TYPE)((((TYPE_DOUBLE_SIZE)left) << FIXED_FBITS) / (TYPE_DOUBLE_SIZE)right));
}

FixedNum fxp_pow2(const FixedNum x){
  if (x == INF_POS)
    return (INF_POS);
  if (x == INF_NEG)
    return ((TYPE)0);

  FixedNum k, f, i;

  if (x < 0){
    k = -x;
    i = (k >> FIXED_FBITS);
    if (k & MASK){
      f = (1 << FIXED_FBITS) - (k & MASK);
      i++;
    }
    else
      f = 0;
  }
  else{
    k = x;
    i = (k >> FIXED_FBITS);
    f = (k & MASK);
  }
	//Serial.println(k&MASK);
	//Serial.println(i);
	//Serial.println(f);

#ifdef SIGNED
  if (i + FIXED_FBITS >= TOTAL_BITS-1)
    return (x > 0 ? (INF_POS) : ((TYPE)0));
#else
  if (i + FIXED_FBITS >= TOTAL_BITS)
    return (x > 0 ? (INF_POS) : ((TYPE)0));
#endif

  FixedNum ans = ONE_FXP;
  if (f > 0)
    ans = fxp_sum(fxp_mul(fxp_sum(fxp_mul(POW_CF1, f), POW_CF2), f), POW_CF3);
//Serial.println(ans);
  if (x > 0)
    ans <<= i;
  else
    ans >>= i;

  return ans;
}

// NEED TO SET LIMITS FOR EACH DATA TYPE
FixedNum fxp_exp(const FixedNum x){
	if (x == INF_POS)
		return (INF_POS);
	if (x == INF_NEG)
		return ((TYPE)0);

	return fxp_pow2(fxp_mul(x, EXP_CF));
}

// SIGNED has to be defined
FixedNum fxp_log2(FixedNum x){
	if (x <= 0)
		return ((TYPE)0);

	FixedNum n = 0;

	while (x < ((TYPE)1 << FIXED_FBITS)){
		n--;
		x <<= 1;
	}

	while (x >= ((TYPE)2 << FIXED_FBITS)){
		n++;
		x >>= 1;
	}

	FixedNum lf = fxp_sum(fxp_mul(fxp_sum(fxp_mul(0xfffffea7, x), 0x000007fb), x), 0xfffff967);

	if (n < 0)
		return (-(abs(n) << FIXED_FBITS) + lf);
	return ((n << FIXED_FBITS) + lf);
}

// FixedNum fxp_pow(FixedNum x, FixedNum y){
// 	return fxp_pow2(fxp_mul(y, fxp_log2(x))); // problema: x < 0
// }

FixedNum fxp_sqrt(FixedNum x){
	if (x <= 0)
		return ((TYPE)0);

	FixedNum n = 0;

	while (x < ((TYPE)1 << FIXED_FBITS)){
		n--;
		x <<= 2;
	}

	while (x >= ((TYPE)4 << FIXED_FBITS)){
		n++;
		x >>= 2;
	}
	// a = -0.0352734056440601
	// b = 0.502292959436962
	// c = 0.546736985360471
	FixedNum lf = fxp_sum(fxp_mul(fxp_sum(fxp_mul(SQRT_CF1, x), SQRT_CF2), x), SQRT_CF3);

	return fxp_mul((1 << (FIXED_FBITS + n)), lf);
}

FixedNum fxp_pow_frac(FixedNum x, FixedNum y){
	if (y == 0) return (1 << FIXED_FBITS);
	FixedNum ans = (1 << FIXED_FBITS);

	for (int i = 0; i < FIXED_FBITS; i++){
		if (y & 1) ans = fxp_mul(ans, x);
		ans = fxp_sqrt(ans);
		y >>= 1;
	}

	return ans;
}

FixedNum fxp_pow_int(FixedNum x, FixedNum y){
	if (y == 0) return (1 << FIXED_FBITS);
	FixedNum ans = (1 << FIXED_FBITS);

	while (y > 0){
		if (y % 2 == 1) ans = fxp_mul(ans, x);
		x = fxp_mul(x, x);
		y >>= 1;
	}

	return ans;
}

FixedNum fxp_pow(FixedNum x, FixedNum y){
	// pow(x,y)==pow(x, floor(y)) * pow(x, frac(y))
	return fxp_mul(fxp_pow_int(x, (y >> FIXED_FBITS)), fxp_pow_frac(x, (y & MASK)));
}

#endif
