package nn

type Euclidean struct {
	target []float64
	x      []float64
	dx     []float64
}

func NewEuclidean(x int) *Euclidean {
	model := &Euclidean{}
	model.dx = make([]float64, x)
	return model
}

func (model *Euclidean) Forward(target, x []float64) float64 {
	model.target = target
	model.x = x
	var loss float64
	for i := 0; i < len(model.x); i++ {
		diff := model.target[i] - model.x[i]
		loss += diff * diff
	}
	return loss
}

func (model *Euclidean) Backward() {
	for i := 0; i < len(model.x); i++ {
		model.dx[i] = -2 * (model.target[i] - model.x[i])
	}
}

func (model *Euclidean) Grad() [1][2][]float64 {
	var grad [1][2][]float64
	grad[0] = [2][]float64{model.x, model.dx}
	return grad
}
