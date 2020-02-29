from ctypes import c_long
from ctypes import Structure
from ctypes import cdll

class sampler_config(Structure):
  _fields_ = [("RngSeed", c_long), ("JobCount", c_long)]


lib = cdll.LoadLibrary("./sampler_main.so")
obj = sampler_config(42, 20)

lib.RunSampler.argtypes = [sampler_config]
lib.RunSampler(obj)
