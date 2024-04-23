# (C) Copyright 2023 European Centre for Medium-Range Weather Forecasts.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
from functools import cached_property
import xarray as xr
import numpy as np
import sys

import climetlab as cml

LOG = logging.getLogger(__name__)


class RequestBasedInput:
    def __init__(self, owner, **kwargs):
        self.owner = owner

    def _patch(self, **kargs):
        r = dict(**kargs)
        self.owner.patch_retrieve_request(r)
        return r

    @cached_property
    def fields_sfc(self):
        LOG.info(f"Loading surface fields from {self.WHERE}")

        return cml.load_source(
            "multi",
            [
                self.sfc_load_source(
                    **self._patch(
                        date=date,
                        time=time,
                        param=self.owner.param_sfc,
                        grid=self.owner.grid,
                        area=self.owner.area,
                        **self.owner.retrieve,
                    )
                )
                for date, time in self.owner.datetimes()
            ],
        )

    @cached_property
    def fields_pl(self):
        LOG.info(f"Loading pressure fields from {self.WHERE}")
        param, level = self.owner.param_level_pl
        return cml.load_source(
            "multi",
            [
                self.pl_load_source(
                    **self._patch(
                        date=date,
                        time=time,
                        param=param,
                        level=level,
                        grid=self.owner.grid,
                        area=self.owner.area,
                    )
                )
                for date, time in self.owner.datetimes()
            ],
        )

    @cached_property
    def all_fields(self):
        return self.fields_sfc + self.fields_pl


class MarsInput(RequestBasedInput):
    WHERE = "MARS"

    def __init__(self, owner, **kwargs):
        self.owner = owner

    def pl_load_source(self, **kwargs):
        kwargs["levtype"] = "pl"
        logging.debug("load source mars %s", kwargs)
        return cml.load_source("mars", kwargs)

    def sfc_load_source(self, **kwargs):
        kwargs["levtype"] = "sfc"
        logging.debug("load source mars %s", kwargs)
        return cml.load_source("mars", kwargs)


class CdsInput(RequestBasedInput):
    WHERE = "CDS"

    def pl_load_source(self, **kwargs):
        kwargs["product_type"] = "reanalysis"
        return cml.load_source("cds", "reanalysis-era5-pressure-levels", kwargs)

    def sfc_load_source(self, **kwargs):
        kwargs["product_type"] = "reanalysis"
        return cml.load_source("cds", "reanalysis-era5-single-levels", kwargs)


class FileInput:
    def __init__(self, owner, file, **kwargs):
        self.file = file
        self.owner = owner

    @cached_property
    def fields_sfc(self):
        date_times = [np.datetime64(f"{str(date)[:4]}-{str(date)[4:6]}-{str(date)[6:]}T{str(time).zfill(2)}") \
                    for date, time in self.owner.datetimes()]
        
        ds = xr.open_dataset(self.file + "_surface.grib")
        if self.owner.post_processing: # if we are doing general post processing, we don't need to keep the
            # following surface fields
            ds_new = ds.sel(time=date_times)
            ds.close()
            del ds
            return ds_new
        ds_times = [ds.time.values + step for step in ds.step.values] # ds.time contains first date only
        steps_idxs = [i for i in range(len(ds_times)) if ds_times[i] in date_times]
        #if len(steps_idxs)==1:
        #    LOG.info([ds_times[steps_idx] for steps_idx in steps_idxs])
        #    LOG.info(date_times)
        ds_new = ds.sel(step=ds.step.values[steps_idxs])
        ds.close()
        del ds
        return ds_new

    @cached_property
    def fields_pl(self):
        date_times = [np.datetime64(f"{str(date)[:4]}-{str(date)[4:6]}-{str(date)[6:]}T{str(time).zfill(2)}") \
                    for date, time in self.owner.datetimes()]
        
        ds = xr.open_dataset(self.file + "_upper.grib")
        #if len(date_times)==2:
            #LOG.info(ds.time.values)
            #LOG.info(date_times)
        ds_new = ds.sel(time=date_times)
        ds.close()
        del ds
        return ds_new

    @cached_property
    def all_fields(self):
        return [self.fields_sfc, self.fields_pl]


INPUTS = dict(
    mars=MarsInput,
    file=FileInput,
    cds=CdsInput,
)


def get_input(name, *args, **kwargs):
    return INPUTS[name](*args, **kwargs)


def available_inputs():
    return sorted(INPUTS.keys())
