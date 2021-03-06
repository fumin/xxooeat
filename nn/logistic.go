package nn

import (
	"math"
)

type Logistic struct {
	x  []float64
	dx []float64
	y  []float64
}

func NewLogistic(x int) *Logistic {
	model := &Logistic{}
	model.dx = make([]float64, x)
	model.y = make([]float64, x)
	return model
}

func (model *Logistic) Forward(x []float64) []float64 {
	model.x = x
	for i := 0; i < len(model.x); i++ {
		model.y[i] = 1 / (1 + math.Exp(-model.x[i]))
	}
	return model.y
}

func (model *Logistic) Backward(dy []float64) {
	for i := 0; i < len(model.dx); i++ {
		model.dx[i] = model.y[i] * (1 - model.y[i]) * dy[i]
	}
}

func (model *Logistic) Grad() [1][2][]float64 {
	var grad [1][2][]float64
	grad[0] = [2][]float64{model.x, model.dx}
	return grad
}
