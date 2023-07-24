package gradientdescent

import (
	"math"
)

type GradientDescent struct {
	learningRate float64
	breakVal     float64
}

func New(learningRate, breakVal float64) *GradientDescent {
	return &GradientDescent{
		learningRate: learningRate,
		breakVal:     breakVal,
	}
}

func (gd *GradientDescent) GradientDescent(points [][]float64) (float64, float64) {
	est := generateEst(points, 0, 0)
	w, b := 0.0, 0.0
	dw, db := math.MaxFloat64, math.MaxFloat64
	for {
		w, b, dw, db = gd.descend(points, est, w, b)
		if math.Abs(dw) < gd.breakVal && math.Abs(db) < gd.breakVal {
			return w, b
		}
		est = generateEst(points, w, b)
	}
}

func generateEst(points [][]float64, w, b float64) [][]float64 {
	var est [][]float64
	for _, point := range points {
		x := point[0]
		est = append(est, []float64{x, w*x + b})
	}
	return est
}

func (gd *GradientDescent) descend(points, est [][]float64, w, b float64) (float64, float64, float64, float64) {
	wTemp := gd.differentiateW(points, est, w, b)
	bTemp := gd.differentiateB(points, est, w, b)
	w -= gd.learningRate * wTemp
	b -= gd.learningRate * bTemp
	return w, b, wTemp, bTemp
}

func (gd *GradientDescent) differentiateW(points, est [][]float64, w, b float64) float64 {
	l := len(points)
	m := float64(l)
	sum := 0.0
	for i := 0; i < l; i++ {
		sum += (est[i][1] - points[i][1]) * est[i][1]
	}
	return sum / m
}

func (gd *GradientDescent) differentiateB(points, est [][]float64, w, b float64) float64 {
	l := len(points)
	m := float64(l)
	sum := 0.0
	for i := 0; i < l; i++ {
		sum += est[i][1] - points[i][1]
	}
	return sum / m
}
