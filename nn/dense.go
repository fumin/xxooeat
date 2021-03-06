package nn

import (
	"log"
	"math/rand"
)

type Dense struct {
	W  []float64
	dw []float64

	B  []float64
	db []float64

	x  []float64
	dx []float64

	y []float64
}

func NewDense(x, y int) *Dense {
	model := &Dense{}

	model.W = make([]float64, x*y)
	for i := 0; i < len(model.W); i++ {
		model.W[i] = rand.Float64()*2 - 1
	}
	model.dw = make([]float64, x*y)

	model.B = make([]float64, y)
	for i := 0; i < len(model.B); i++ {
		model.B[i] = rand.Float64()*2 - 1
	}
	model.db = make([]float64, y)

	model.dx = make([]float64, x)
	model.y = make([]float64, y)

	return model
}

func (model *Dense) Forward(x []float64) []float64 {
	model.x = x
	for i := 0; i < len(model.y); i++ {
		model.y[i] = model.B[i]
		for j := 0; j < len(model.x); j++ {
			model.y[i] += model.W[i*len(model.x)+j] * x[j]
		}
	}
	return model.y
}

func (model *Dense) Backward(dy []float64) {
	// dw
	for i := 0; i < len(model.y); i++ {
		for j := 0; j < len(model.x); j++ {
			model.dw[i*len(model.x)+j] = dy[i] * model.x[j]
		}
	}

	// db
	for i := 0; i < len(model.y); i++ {
		model.db[i] = dy[i]
	}

	// dx
	for j := 0; j < len(model.x); j++ {
		model.dx[j] = 0
		for i := 0; i < len(model.y); i++ {
			model.dx[j] += model.W[i*len(model.x)+j] * dy[i]
		}
	}
}

func (model *Dense) Grad() [3][2][]float64 {
	var grad [3][2][]float64
	grad[0] = [2][]float64{model.x, model.dx}
	grad[1] = [2][]float64{model.W, model.dw}
	grad[2] = [2][]float64{model.B, model.db}
	return grad
}

func denselog() {
	log.Printf("")
}
