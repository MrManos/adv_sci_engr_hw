package main

import (
	"fmt"
	"math"
)

// function f(x) = x^3 - 3x - 3
func f(x float64) float64 {
	return math.Pow(x, 3) - 3*x - 3
}

// function f'(x) = 3x^2 - 3
func f_prime(x float64) float64 {
	return 3*math.Pow(x, 2) - 3
}

func newtonsMethod(x0 float64, iterations int) float64 {
	x := x0
	for i := 0; i < iterations; i++ {
		x = x - f(x)/f_prime(x)
		fmt.Printf("Iteration %d: x = %.6f\n", i+1, x)
	}
	return x
}

func main() {

	fmt.Println("\nNewton's method to find x-intercept near x = 2:")
	initialGuess := 2.0
	iterations := 2
	xIntercept := newtonsMethod(initialGuess, iterations)
	fmt.Printf("Approximate x-intercept: %.6f\n", xIntercept)
}
