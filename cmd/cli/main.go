package main

import (
	"bufio"
	"flag"
	"fmt"
	"log"
	"math"
	"os"
	"strconv"
	"strings"

	"github.com/fumin/xxooeat"
	"github.com/fumin/xxooeat/model"
	"github.com/pkg/errors"
)

func main() {
	flag.Parse()
	log.SetFlags(log.LstdFlags | log.Lmicroseconds | log.Lshortfile)

	var player0 Agent = Human{}
	var player1 Agent = Human{}
	var err error
	player0, err = NewMLP("/Users/mac/fumin/tmp/xxooeat/td128_128/model")
	if err != nil {
		log.Fatalf("%+v", err)
	}
	player1, err = NewMLP("/Users/mac/fumin/tmp/xxooeat/td128_128/model")
	if err != nil {
		log.Fatalf("%+v", err)
	}
	if err := play(3, 3, 2, []Agent{player0, player1}); err != nil {
		log.Fatalf("%+v", err)
	}
}

func play(boardSize, maxPieceSize, numPiecesPerSize int, players []Agent) error {
	state := xxooeat.NewState(boardSize, maxPieceSize, numPiecesPerSize)
	state.CanMove = true
	var step int
	for {
		fmt.Printf("Step %d:\n", step)
		fmt.Printf("%s\n", state)
		actions := state.Next()
		if len(actions) == 0 {
			fmt.Printf("Draw.\n")
			return nil
		}

		if _, ok := players[state.Player()].(Human); ok {
			fmt.Printf("You are player %d, and your available actions are: \n", state.Player())
			for i, act := range actions {
				fmt.Printf("%d\n", i)
				fmt.Printf("%s\n", act.State)
			}
		}

		actIdx, err := players[state.Player()].Act(actions)
		if err != nil {
			return errors.Wrap(err, "")
		}
		state = actions[actIdx].State
		win := state.Win()
		switch {
		case win[0] && win[1]:
			fmt.Printf("%s\n", state)
			fmt.Printf("Draw\n")
			return nil
		case win[0]:
			fmt.Printf("%s\n", state)
			fmt.Printf("Player 0 Wins.\n")
			return nil
		case win[1]:
			fmt.Printf("%s\n", state)
			fmt.Printf("Player 1 Wins.\n")
			return nil
		}
		fmt.Printf("\n")

		step++
	}
}

type Agent interface {
	Act(actions []xxooeat.Action) (int, error)
}

type MLP struct {
	valueNet *model.MLP
}

func NewMLP(fpath string) (*MLP, error) {
	agent := &MLP{}
	b, err := os.ReadFile(fpath)
	if err != nil {
		return nil, errors.Wrap(err, "")
	}
	agent.valueNet, err = model.LoadMLP(b)
	if err != nil {
		return nil, errors.Wrap(err, "")
	}
	return agent, nil
}

func (agent *MLP) Act(actions []xxooeat.Action) (int, error) {
	var maxIdx int = -1
	var maxValue float64 = -math.MaxFloat64
	for i, act := range actions {
		_, value := agent.valueNet.Forward([]float64{-1}, act.State.S)
		if value > maxValue {
			maxIdx = i
			maxValue = value
		}
		stateStr := act.State.String()
		stateStr = strings.ReplaceAll(stateStr, "\n", "\n\t")
		fmt.Printf("\t%d agent %f\n\t%s\n", i, value, stateStr)
	}
	return maxIdx, nil
}

type Human struct{}

func (human Human) Act(actions []xxooeat.Action) (int, error) {
	scanner := bufio.NewScanner(os.Stdin)
	var actIdx int = -1
	for {
		if !scanner.Scan() {
			return -1, scanner.Err()
		}
		actStr := scanner.Text()
		var err error
		actIdx, err = strconv.Atoi(actStr)
		if err != nil {
			fmt.Printf("wrong action: %s\n", actStr)
			continue
		}
		if actIdx >= len(actions) {
			fmt.Printf("action \"%s\" must be smaller than %d\n", actStr, len(actions))
			continue
		}
		break
	}
	return actIdx, nil
}
