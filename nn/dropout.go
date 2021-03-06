package nn

import (
	"log"
	"math/rand"
)

type Dropout struct {
	p       float64
	x       []float64
	dx      []float64
	dropped []bool
	y       []float64
}

func NewDropout(p float64, x int) *Dropout {
	model := &Dropout{}
	if p <= 0 || p >= 1 {
		log.Fatalf("%f", p)
	}
	model.p = p
	model.dx = make([]float64, x)
	model.dropped = make([]bool, x)
	model.y = make([]float64, x)
	return model
}

func (model *Dropout) Forward(x []float64) []float64 {
	model.x = x
	multiplier := 1.0 / (1 - model.p)
	for i := 0; i < len(model.dropped); i++ {
		if rand.Float64() < model.p {
			model.dropped[i] = true
			model.y[i] = 0
		} else {
			model.dropped[i] = false
			model.y[i] = x[i] * multiplier
		}
	}
	return model.y
}

func (model *Dropout) Backward(dy []float64) {
	multiplier := 1.0 / (1 - model.p)
	for i := 0; i < len(model.dropped); i++ {
		if model.dropped[i] {
			model.dx[i] = 0
		} else {
			model.dx[i] = dy[i] * multiplier
		}
	}
}

func (model *Dropout) Grad() [1][2][]float64 {
	var grad [1][2][]float64
	grad[0] = [2][]float64{model.x, model.dx}
	return grad
}
