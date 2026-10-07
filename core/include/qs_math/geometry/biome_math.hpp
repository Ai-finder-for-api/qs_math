#pragma once
// qs_math/biome_math.hpp — Koppen classification, vegetation index, ecosystem dynamics (C++20)
#include "scalar_types.hpp"
#include "vec2.hpp"
#include "vec3.hpp"
#include "random.hpp"
#include <vector>
#include <cmath>
#include <algorithm>
#include <array>
#include <string>

namespace qs {

// ─── Climate data ─────────────────────────────────────────────────────────────

struct ClimateCell {
    f32 mean_temp{15.f};         // degrees C
    f32 temp_range{20.f};        // annual range
    f32 annual_precip{800.f};    // mm/year
    f32 driest_month_precip{30.f};
    f32 wettest_month_precip{120.f};
    f32 warmest_month_temp{25.f};
    f32 coldest_month_temp{5.f};
    f32 elevation{0.f};          // meters
    f32 latitude{45.f};          // degrees
};

// ─── Koppen climate classification ────────────────────────────────────────────

enum class KoppenClass {
    Af,  // Tropical rainforest
    Am,  // Tropical monsoon
    Aw,  // Tropical savanna
    BSh, // Hot semi-arid
    BSk, // Cold semi-arid
    BWh, // Hot desert
    BWk, // Cold desert
    Cfa, // Humid subtropical
    Cfb, // Oceanic
    Cfc, // Subpolar oceanic
    Csa, // Mediterranean hot summer
    Csb, // Mediterranean warm summer
    Cwa, // Monsoon humid subtropical
    Dfa, // Humid continental hot summer
    Dfb, // Humid continental warm summer
    Dfc, // Subarctic
    Dfd, // Extremely cold subarctic
    Dwa, // Monsoon humid continental hot
    Dwb, // Monsoon humid continental warm
    Dwc, // Monsoon subarctic
    ET,  // Tundra
    EF,  // Ice cap
    Unknown
};

[[nodiscard]] inline KoppenClass koppen_classify(const ClimateCell& c) {
    // Ice cap
    if(c.warmest_month_temp<0.f) return KoppenClass::EF;
    // Tundra
    if(c.warmest_month_temp<10.f) return KoppenClass::ET;

    f32 P=c.annual_precip, T=c.mean_temp;
    // Threshold for arid climates
    f32 P_thresh;
    if(c.wettest_month_precip>c.driest_month_precip*3.f)
        P_thresh=2.f*(T+14.f); // winter dry
    else if(c.wettest_month_precip>c.driest_month_precip*10.f)
        P_thresh=2.f*(T+7.f);  // summer dry
    else
        P_thresh=2.f*T;

    bool arid=P<10.f*P_thresh;
    if(arid) {
        bool hot=T>18.f;
        if(P<5.f*P_thresh) return hot?KoppenClass::BWh:KoppenClass::BWk;
        return hot?KoppenClass::BSh:KoppenClass::BSk;
    }

    bool tropical=c.coldest_month_temp>=18.f;
    if(tropical){
        if(c.driest_month_precip>=60.f) return KoppenClass::Af;
        if(c.driest_month_precip>=100.f-P/25.f) return KoppenClass::Am;
        return KoppenClass::Aw;
    }

    bool summer_dry=c.driest_month_precip<30.f&&c.driest_month_precip<c.wettest_month_precip/3.f;
    bool mild_winter=c.coldest_month_temp>-3.f;

    if(mild_winter) {
        if(summer_dry) return c.warmest_month_temp>22.f?KoppenClass::Csa:KoppenClass::Csb;
        if(c.coldest_month_temp<18.f&&c.warmest_month_temp>22.f) return KoppenClass::Cfa;
        return KoppenClass::Cfb;
    }
    // Continental
    if(c.warmest_month_temp>22.f) return KoppenClass::Dfa;
    if(c.coldest_month_temp>-38.f) return KoppenClass::Dfb;
    if(c.coldest_month_temp>-38.f) return KoppenClass::Dfc;
    return KoppenClass::Dfd;
}

[[nodiscard]] inline const char* koppen_name(KoppenClass k) {
    switch(k){
    case KoppenClass::Af: return "Tropical Rainforest";
    case KoppenClass::Am: return "Tropical Monsoon";
    case KoppenClass::Aw: return "Tropical Savanna";
    case KoppenClass::BSh:return "Hot Semi-arid";
    case KoppenClass::BWh:return "Hot Desert";
    case KoppenClass::BWk:return "Cold Desert";
    case KoppenClass::Cfa:return "Humid Subtropical";
    case KoppenClass::Cfb:return "Oceanic";
    case KoppenClass::Csa:return "Mediterranean Hot Summer";
    case KoppenClass::Dfa:return "Humid Continental Hot";
    case KoppenClass::Dfb:return "Humid Continental Warm";
    case KoppenClass::Dfc:return "Subarctic";
    case KoppenClass::ET: return "Tundra";
    case KoppenClass::EF: return "Ice Cap";
    default:              return "Unknown";
    }
}

// ─── Biome from Whittaker (temp/precip) ──────────────────────────────────────

enum class WhittakerBiome {
    TropicalRainforest, TropicalDryForest, Savanna, Desert,
    TemperateRainforest, TemperateForest, Shrubland,
    Grassland, Boreal, Tundra, Wetland
};

[[nodiscard]] inline WhittakerBiome whittaker_biome(f32 mean_temp_C, f32 annual_precip_mm) {
    if(mean_temp_C>20&&annual_precip_mm>2000) return WhittakerBiome::TropicalRainforest;
    if(mean_temp_C>20&&annual_precip_mm>1000) return WhittakerBiome::TropicalDryForest;
    if(mean_temp_C>20&&annual_precip_mm>500)  return WhittakerBiome::Savanna;
    if(annual_precip_mm<250)                  return WhittakerBiome::Desert;
    if(mean_temp_C>10&&annual_precip_mm>2000) return WhittakerBiome::TemperateRainforest;
    if(mean_temp_C>10&&annual_precip_mm>750)  return WhittakerBiome::TemperateForest;
    if(mean_temp_C>5&&annual_precip_mm>300)   return WhittakerBiome::Shrubland;
    if(mean_temp_C>5)                         return WhittakerBiome::Grassland;
    if(mean_temp_C>-5)                        return WhittakerBiome::Boreal;
    return WhittakerBiome::Tundra;
}

// ─── NDVI and vegetation indices ─────────────────────────────────────────────

[[nodiscard]] inline f32 ndvi(f32 nir, f32 red) {
    f32 d=nir+red; return d>0.f?(nir-red)/d:0.f;
}

[[nodiscard]] inline f32 evi(f32 nir, f32 red, f32 blue,
    f32 G=2.5f, f32 C1=6.f, f32 C2=7.5f, f32 L=1.f)
{
    f32 d=nir+C1*red-C2*blue+L;
    return d>0.f?G*(nir-red)/d:0.f;
}

[[nodiscard]] inline f32 savi(f32 nir, f32 red, f32 L_soil=0.5f) {
    return (nir-red)*(1.f+L_soil)/(nir+red+L_soil);
}

[[nodiscard]] inline f32 lai_from_ndvi(f32 ndvi_val) {
    // Empirical relationship (Beer-Lambert approximation)
    return -std::log(std::max(0.01f,1.f-ndvi_val))/0.5f;
}

// ─── Ecosystem carbon model (simple) ─────────────────────────────────────────

struct EcosystemCarbon {
    f32 npp{};         // net primary productivity (gC/m²/yr)
    f32 soil_carbon{}; // soil organic carbon pool (gC/m²)
    f32 biomass{};     // aboveground biomass (gC/m²)
    f32 litter{};      // litter pool (gC/m²)

    struct Params {
        f32 k_litter{0.5f};   // litter decomposition rate (/yr)
        f32 k_soil{0.05f};    // soil decomp rate
        f32 CUE{0.4f};        // carbon use efficiency
        f32 litter_fraction{0.3f}; // of NPP to litter
    };

    void step(f32 dt, const Params& p) {
        f32 litter_in=npp*p.litter_fraction;
        f32 litter_decomp=litter*p.k_litter*dt;
        f32 soil_in=litter_decomp*p.CUE;
        f32 soil_decomp=soil_carbon*p.k_soil*dt;
        biomass+=npp*(1.f-p.litter_fraction)*dt;
        litter+=litter_in*dt-litter_decomp;
        soil_carbon+=soil_in-soil_decomp;
        litter=std::max(0.f,litter);
        soil_carbon=std::max(0.f,soil_carbon);
    }

    [[nodiscard]] f32 total_carbon() const { return biomass+litter+soil_carbon; }
    [[nodiscard]] f32 nep() const { return npp-soil_carbon*0.05f-litter*0.5f; } // net ecosystem productivity
};

// ─── Species diversity (Shannon / Simpson) ────────────────────────────────────

[[nodiscard]] inline f32 shannon_diversity(std::span<const f32> abundances) {
    f32 total=0.f; for(f32 a:abundances) total+=a;
    if(total<=0.f) return 0.f;
    f32 H=0.f;
    for(f32 a:abundances){
        f32 p=a/total;
        if(p>0.f) H-=p*std::log(p);
    }
    return H;
}

[[nodiscard]] inline f32 simpson_diversity(std::span<const f32> abundances) {
    f32 total=0.f,sum=0.f;
    for(f32 a:abundances){total+=a;sum+=a*a;}
    return total>0.f?1.f-sum/(total*total):0.f;
}

[[nodiscard]] inline f32 species_richness(std::span<const f32> abundances, f32 threshold=0.001f) {
    u32 r=0; for(f32 a:abundances) if(a>threshold) ++r;
    return f32(r);
}

// Lotka-Volterra (2-species predator-prey)
struct LotkaVolterra {
    f32 prey{100.f}, pred{10.f};
    f32 alpha{1.f};   // prey birth rate
    f32 beta{0.1f};   // predation rate
    f32 gamma{0.075f};// predator birth rate from prey
    f32 delta{1.5f};  // predator death rate

    void step(f32 dt){
        f32 dprey=(alpha*prey-beta*prey*pred)*dt;
        f32 dpred=(gamma*prey*pred-delta*pred)*dt;
        prey=std::max(0.f,prey+dprey);
        pred=std::max(0.f,pred+dpred);
    }
};

// ─── Atmospheric CO2 / temperature feedback ───────────────────────────────────

[[nodiscard]] inline f32 radiative_forcing_co2(f32 co2_ppm, f32 co2_ref=280.f) {
    // IPCC simplified formula: 5.35 * ln(C/C0) W/m²
    return 5.35f*std::log(co2_ppm/co2_ref);
}

[[nodiscard]] inline f32 global_temp_from_forcing(f32 forcing_W_m2, f32 climate_sensitivity=3.f) {
    // ECS: degrees per doubling of CO2 (default ~3C)
    return forcing_W_m2*climate_sensitivity/3.71f; // 3.71 W/m² per doubling
}

} // namespace qs
