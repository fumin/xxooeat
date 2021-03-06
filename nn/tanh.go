package nn

import (
	"math"
)

type Tanh struct {
	x  []float64
	dx []float64
	y  []float64
}

func NewTanh(x int) *Tanh {
	model := &Tanh{}
	model.dx = make([]float64, x)
	model.y = make([]float64, x)
	return model
}

func (model *Tanh) Forward(x []float64) []float64 {
	model.x = x
	for i := 0; i < len(model.x); i++ {
		model.y[i] = math.Tanh(model.x[i])
	}
	return model.y
}

func (model *Tanh) Backward(dy []float64) {
	for i := 0; i < len(model.dx); i++ {
		model.dx[i] = (1 - model.y[i]*model.y[i]) * dy[i]
	}
}

func (model *Tanh) Grad() [1][2][]float64 {
	var grad [1][2][]float64
	grad[0] = [2][]float64{model.x, model.dx}
	return grad
}
