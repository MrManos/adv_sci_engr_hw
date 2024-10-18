package main

import (
	"fmt"
	"math"
)

// Function f(x) = 4 + 8x^2 - x^4
func f(x float64) float64 {
	return 4 + 8*x*x - math.Pow(x, 4)
}

// First derivative f'(x) = 16x - 4x^3
func f_prime(x float64) float64 {
	return 16*x - 4*math.Pow(x, 3)
}

// Newton's method for finding x-intercept
func newtonsMethod(x0 float64, iterations int) float64 {
	x := x0
	for i := 0; i < iterations; i++ {
		x = x - f(x)/f_prime(x)
		fmt.Printf("Iteration %d: x = %.6f\n", i+1, x)
	}
	return x
}

func main() {

	// Part 4: Newton's Method for x-intercept near x = 3
	fmt.Println("\nNewton's method to find x-intercept near x = 3:")
	initialGuess := 3.0
	iterations := 2
	xIntercept := newtonsMethod(initialGuess, iterations)
	fmt.Printf("Approximate x-intercept: %.6f\n", xIntercept)
}
