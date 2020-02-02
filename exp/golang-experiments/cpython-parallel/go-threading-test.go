// Tester for multithreaded go run in python
package main

// Comment in import is the C definition used of the python object passed in
import (
	/*
		typedef struct {
			long t1;
			long t2;
		}obj;
	*/
	"C"
	"fmt"
	"sync"
)

//export Main
func Main(testObj C.obj) {

	// Replace with recurseFib for non-parallel version
	testObj.t1 = recurseFibParallel(testObj.t1)
	testObj.t2 = recurseFibParallel(testObj.t2)
	fmt.Println(testObj)
}

// Recursively solves for the nth fib number in paralell
func recurseFibParallel(val C.long) C.long {
	if val == 1 || val == 0 {
		return val
	}

	var val1 C.long
	var val2 C.long
	var wg sync.WaitGroup
	wg.Add(2)

	go func() {
		val1 = recurseFib(val - 1)
		wg.Done()
	}()
	go func() {
		val2 = recurseFib(val - 2)
		wg.Done()
	}()
	wg.Wait()
	return val1 + val2
}

// Recursively solves for the nth fib number not in parallel
func recurseFib(val C.long) C.long {
	if val == 1 || val == 0 {
		return val
	}

	var val1 C.long
	var val2 C.long
	val1 = recurseFib(val - 1)
	val2 = recurseFib(val - 2)
	return val1 + val2
}

func main() {}
