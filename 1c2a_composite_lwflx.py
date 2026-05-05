#to run in terminal paste: sbatch --job-name=lwflx --export=SCRIPT=1c2a_composite_lwflx.py submit_composite.sh

import xarray as xr
import xesmf as xe
import numpy as np
import sys
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)

# ── Configuration ────────────────────────────────────────────────────────────
formatted_list = [str(i).zfill(2) for i in range(1, 31)]
TIME_START = '1991'
TIME_END   = '2024'
BASE_PATH  = '/work/stb/MHW-gfdl/SPEAR/SPEAR_c384_OM4p25_Hist_SSP245_IC1991_R61_ens_{e}'
MASK_PATH  = '/work/stb/MHW-gfdl/SPEAR/vars/SPEAR-HI25/period_1991-2024/events/event_mask_sphi{e}.nc'
SAVE_PATH  = '/work/stb/MHW-gfdl/SPEAR/vars/SPEAR-HI25/period_1991-2024/composites/ens-mean/'
HF_VAR     = 'lwflx'

# ── Load heat flux ensemble ───────────────────────────────────────────────────
def preprocess_flux(ds):
    return ds[[HF_VAR]].sel(lat=slice(10, 31), lon=slice(262, 344), time=slice(TIME_START, TIME_END))

print("Opening heat flux ensemble...")
flux_ds = xr.open_mfdataset(
    [f'{BASE_PATH.format(e=e)}/{HF_VAR}_ens{e}_1991-2100.nc' for e in formatted_list],
    preprocess=preprocess_flux,
    concat_dim='ensemble',
    combine='nested',
    coords='minimal',
    compat='override',
    decode_timedelta=True,
    chunks={'ensemble': 1, 'time': -1, 'lat': -1, 'lon': -1},
)
flux_ds = flux_ds.assign_coords(ensemble=[int(e) for e in formatted_list])

# ── Load MHW mask ensemble ────────────────────────────────────────────────────
print("Opening MHW mask ensemble...")
mask_ds_list = []
for e in formatted_list:
    ds = xr.open_mfdataset(MASK_PATH.format(e=e), decode_timedelta=True).rename_vars({'__xarray_dataarray_variable__': 'event_mask'})
    mask_ds_list.append(ds)

mask_ds = xr.concat(mask_ds_list, dim='ensemble')
mask_ds = mask_ds.assign_coords(ensemble=[int(e) for e in formatted_list])

# ── Build regridder once using member 0 as template ──────────────────────────
print("Building regridder...")
flux_template = flux_ds[HF_VAR].isel(ensemble=0, time=0).compute()
mask_template = mask_ds.event_mask.isel(ensemble=0, time=0).compute()
regridder = xe.Regridder(flux_template, mask_template, method='bilinear')

# ── Composite loop — one ensemble member at a time ───────────────────────────
results_mhw       = []
results_nomhw     = []
results_clim      = []
results_anom_mhw  = []
results_anom_mhw_season  = []
results_anom_nomhw       = []
results_anom_nomhw_season = []

for i in range(30):
    print(f"Processing member {i+1}/30...")

    flux        = flux_ds[HF_VAR].isel(ensemble=i).compute()
    mask_member = mask_ds.event_mask.isel(ensemble=i).compute()

    # Regrid flux to mask's native ocean grid
    flux_ng = regridder(flux)

    # Align time axes
    flux_ng, mask_aligned = xr.align(flux_ng, mask_member, join='inner')

    # MHW / non-MHW time-mean composites
    mhw_mean   = flux_ng.where(mask_aligned).mean('time')
    nomhw_mean = flux_ng.where(~mask_aligned).mean('time')

    # Daily climatology and MHW anomaly
    clim     = flux_ng.groupby('time.dayofyear').mean('time')
    anom     = flux_ng.groupby('time.dayofyear') - clim
    anom_mhw = anom.where(mask_aligned).mean('time')
    anom_mhw_season = anom.where(mask_aligned).groupby('time.season').mean('time')
    anom_nomhw = anom.where(~mask_aligned).mean('time')
    anom_nomhw_season = anom.where(~mask_aligned).groupby('time.season').mean('time')
    
    results_mhw.append(mhw_mean.drop_vars('ensemble', errors='ignore')) #ignore instances where ensemble isn't a coord
    results_nomhw.append(nomhw_mean.drop_vars('ensemble', errors='ignore'))
    results_clim.append(clim.drop_vars('ensemble', errors='ignore'))
    results_anom_mhw.append(anom_mhw.drop_vars('ensemble', errors='ignore'))
    results_anom_mhw_season.append(anom_mhw_season.drop_vars('ensemble', errors='ignore'))
    results_anom_nomhw.append(anom_nomhw.drop_vars('ensemble', errors='ignore'))
    results_anom_nomhw_season.append(anom_nomhw_season.drop_vars('ensemble', errors='ignore'))

    del flux, flux_ng, mask_member, mask_aligned, clim, anom, anom_mhw, anom_mhw_season, anom_nomhw, anom_nomhw_season # free memory

# ── Ensemble mean of time-composites ─────────────────────────────────────────
print("Computing ensemble means...")
flux_mhw_ens_mean      = xr.concat(results_mhw,      dim='ensemble').mean('ensemble')
flux_nomhw_ens_mean    = xr.concat(results_nomhw,    dim='ensemble').mean('ensemble')
diff_ens_mean_flux     = flux_mhw_ens_mean - flux_nomhw_ens_mean
flux_clim_ens_mean     = xr.concat(results_clim,     dim='ensemble').mean('ensemble')
flux_anom_mhw_ens_mean = xr.concat(results_anom_mhw, dim='ensemble').mean('ensemble')
flux_anom_mhw_season_ens_mean = xr.concat(results_anom_mhw_season, dim='ensemble').mean('ensemble')
flux_anom_nomhw_ens_mean = xr.concat(results_anom_nomhw, dim='ensemble').mean('ensemble')
flux_anom_nomhw_season_ens_mean = xr.concat(results_anom_nomhw_season, dim='ensemble').mean('ensemble')

# ── Save outputs ──────────────────────────────────────────────────────────────
print("Saving outputs...")
flux_mhw_ens_mean.rename('flux_mhw_ens_mean').to_netcdf(SAVE_PATH + f'{HF_VAR}_mhw_composite.nc', mode='w')
flux_nomhw_ens_mean.rename('flux_nomhw_ens_mean').to_netcdf(SAVE_PATH + f'{HF_VAR}_nomhw_composite.nc', mode='w')
diff_ens_mean_flux.rename('diff_ens_mean_flux').to_netcdf(SAVE_PATH + f'{HF_VAR}_diff_composite.nc', mode='w')
flux_clim_ens_mean.rename('flux_clim_ens_mean').to_netcdf(SAVE_PATH + f'{HF_VAR}_clim_ens_mean.nc', mode='w')
print("Saving flux_anom_mhw_ens_mean...")
flux_anom_mhw_ens_mean.rename('flux_anom_mhw_ens_mean').to_netcdf(SAVE_PATH + f'{HF_VAR}_mhw_anom_composite.nc', mode='w')
print("Saving flux_anom_mhw_season_ens_mean...")
flux_anom_mhw_season_ens_mean.rename('flux_anom_mhw_season_ens_mean').to_netcdf(SAVE_PATH + f'{HF_VAR}_mhw_anom_season_composite.nc', mode='w')
print("Saving flux_anom_nomhw_ens_mean...")
flux_anom_nomhw_ens_mean.rename('flux_anom_nomhw_ens_mean').to_netcdf(SAVE_PATH + f'{HF_VAR}_nomhw_anom_composite.nc', mode='w')
print("Saving flux_anom_nomhw_season_ens_mean...")
flux_anom_nomhw_season_ens_mean.rename('flux_anom_nomhw_season_ens_mean').to_netcdf(SAVE_PATH + f'{HF_VAR}_nomhw_anom_season_composite.nc', mode='w')

print("Done.")
