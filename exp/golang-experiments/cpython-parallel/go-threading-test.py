from ctypes import c_long
from ctypes import Structure
from ctypes import cdll

class TestClass(Structure):
  _fields_ = [("t1", c_long), ("t2", c_long)]


lib = cdll.LoadLibrary("./go-threading-test.so")
obj = TestClass(100, 100)

lib.Main.argtypes = [TestClass]
lib.Main(obj)
