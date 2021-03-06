package nn

import (
	"log"
	"math/rand"
)

type Dense struct {
	w  []float64
	dw []float64

	b  []float64
	db []float64

	x  []float64
	dx []float64

	y []float64
}

func NewDense(x, y int) *Dense {
	model := &Dense{}

	model.w = make([]float64, x*y)
	for i := 0; i < len(model.w); i++ {
		model.w[i] = rand.Float64()
	}
	model.dw = make([]float64, x*y)

	model.b = make([]float64, y)
	for i := 0; i < len(model.b); i++ {
		model.b[i] = rand.Float64()
	}
	model.db = make([]float64, y)

	model.dx = make([]float64, x)
	model.y = make([]float64, y)

	return model
}

func (model *Dense) Forward(x []float64) []float64 {
	model.x = x
	for i := 0; i < len(model.y); i++ {
		model.y[i] = model.b[i]
		for j := 0; j < len(model.x); j++ {
			model.y[i] += model.w[i*len(model.x)+j] * x[j]
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
			model.dx[j] += model.w[i*len(model.x)+j] * dy[i]
		}
	}
}

func (model *Dense) Grad() [3][2][]float64 {
	var grad [3][2][]float64
	grad[0] = [2][]float64{model.x, model.dx}
	grad[1] = [2][]float64{model.w, model.dw}
	grad[2] = [2][]float64{model.b, model.db}
	return grad
}

func denselog() {
	log.Printf("")
}
