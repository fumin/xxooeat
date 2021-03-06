package nn

type Sum struct {
	x  []float64
	dx []float64
	y  []float64
}

func NewSum(x int) *Sum {
	model := &Sum{}
	model.dx = make([]float64, x)
	model.y = make([]float64, 1)
	return model
}

func (model *Sum) Forward(x []float64) []float64 {
	model.x = x
	model.y[0] = 0
	for i := 0; i < len(x); i++ {
		model.y[0] += x[i]
	}
	return model.y
}

func (model *Sum) Backward(dy []float64) {
	for i := 0; i < len(model.x); i++ {
		model.dx[i] = dy[0]
	}
}

func (model *Sum) Grad() [1][2][]float64 {
	var grad [1][2][]float64
	grad[0] = [2][]float64{model.x, model.dx}
	return grad
}
