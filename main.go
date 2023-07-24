package main

import (
	"fmt"

	"github.com/aclander/gradient-descent/pkg/gradientdescent"
)

func main() {
	gd := gradientdescent.New(.0001, .000001)
	points := [][]float64{{1.0, 3.0}, {3.0, 7.0}, {4.0, 5.0}, {3.0, 12.0}, {6.0, 6.0}, {8.0, 15.0}}
	w, b := gd.GradientDescent(points)
	fmt.Println("For input:")
	fmt.Println(points)
	fmt.Println("\nBest fit:")
	fmt.Printf("f(x) = %fx + %f\n", w, b)
}
