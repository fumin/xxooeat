package nn

type Relu struct {
	x  []float64
	dx []float64
	y  []float64
}

func NewRelu(x int) *Relu {
	model := &Relu{}
	model.dx = make([]float64, x)
	model.y = make([]float64, x)
	return model
}

func (model *Relu) Forward(x []float64) []float64 {
	model.x = x
	for i := 0; i < len(model.x); i++ {
		if model.x[i] > 0 {
			model.y[i] = x[i]
		} else {
			model.y[i] = 0
		}
	}
	return model.y
}

func (model *Relu) Backward(dy []float64) {
	for i := 0; i < len(model.x); i++ {
		if model.x[i] > 0 {
			model.dx[i] = dy[i]
		} else {
			model.dx[i] = 0
		}
	}
}

func (model *Relu) Grad() [1][2][]float64 {
	var grad [1][2][]float64
	grad[0] = [2][]float64{model.x, model.dx}
	return grad
}
