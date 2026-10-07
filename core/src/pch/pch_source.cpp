// pch_source.cpp — PCH compilation unit for qs_build /Yc
// qs_build passes this file to cl.exe with /Yc"qs_math_pch.hpp"
// to produce the precompiled header. All other TUs consume it via /Yu.
#include "qs_math_pch.hpp"
