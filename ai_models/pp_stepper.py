# A. Leclerc, 18th April 2024toggle blo
# Modification of class Stepper to compute more efficiently
# forecasts needed to train post-processing algorithms.

import logging
import time

from climetlab.utils.humanize import seconds

LOG = logging.getLogger(__name__)
    

class PPStepper:
    def __init__(self, steps, lead_times):
        
        # Checking whether or not the steps are in increasing order
        if not(all(steps[i] < steps[i+1] for i in range(len(steps)-1))):
            raise ValueError("Steps must be in increasing order.")
        
        self.steps = steps #possible steps
        self.lead_times = lead_times #required lead times to compute
        self.start = time.time()
        self.last = self.start
        
        #Dividing the lead times with steps, creating a dict with lead times
        # as keys and a stepping based on steps
        self.stepping = self.PPstepping(steps, lead_times)
        
        self.num_steps = sum(len(self.stepping[key]) for key in self.stepping)
        LOG.info("Starting inference for %s steps.",
                 self.num_steps)
    
    def __enter__(self):
        return self
    
    def __call__(self, i, step):
        now = time.time()
        elapsed = now - self.start
        speed = (i + 1) / elapsed
        eta = (self.num_steps - i) / speed
        LOG.info(
            "Done %s out of %s in %s (%sh), ETA: %s.",
            i + 1,
            self.num_steps,
            seconds(now - self.last),
            step,
            seconds(eta),
        )
        self.last = now
    
    def __exit__(self, *args):
        elapsed = time.time() - self.start
        LOG.info("Elapsed: %s.", seconds(elapsed))
        LOG.info("Average: %s per step.", seconds(elapsed / self.num_steps))
    
    def PPstepping(self, steps, lead_times):
        """
        This function computes the different steps needed to
        get to lead_time with PanguWeather.
        """
        res = {}
        for lead_time in lead_times:
            mem = 0
            lt = lead_time
            res_lt = [0]
            for step in steps[::-1]:
                while lead_time > 0 and lead_time >= step:
                    res_lt.append(step + mem)
                    mem = res_lt[-1]
                    lead_time -= step
            res[lt] = res_lt
        return res
    