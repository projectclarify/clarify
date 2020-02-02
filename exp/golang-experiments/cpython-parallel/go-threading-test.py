from ctypes import c_long
from ctypes import Structure
from ctypes import cdll

class test_class(Structure):
  _fields_ = [("t1", c_long), ("t2", c_long)]


lib = cdll.LoadLibrary("./go-threading-test.so")
obj = test_class(100, 100)

lib.Main.argtypes = [test_class]
lib.Main(obj)
