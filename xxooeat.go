package xxooeat

import (
	"fmt"
	"log"
	"strings"
)

const (
	numPlayers = 2

	PlayerEmpty = -1
	levelEmpty  = -1

	// Implementation detail.
	bitPieceIsPlayed = 0
	bitPiecePlayer   = 1
)

type State struct {
	S                []float64
	BoardSize        int
	MaxPieceSize     int
	NumPiecesPerSize int

	LenPiece         int
	CanMove          bool
	lenTower         int
	bitCurrentPlayer int
}

func NewState(boardSize, maxPieceSize, numPiecesPerSize int) *State {
	s := &State{}
	s.BoardSize = boardSize
	s.MaxPieceSize = maxPieceSize
	s.NumPiecesPerSize = numPiecesPerSize
	s.LenPiece = 4
	s.CanMove = true
	s.lenTower = s.LenPiece * s.MaxPieceSize

	numPositions := boardSize * boardSize
	lenState := s.lenTower*numPositions + 1
	s.S = make([]float64, lenState)

	s.bitCurrentPlayer = lenState - 1
	return s
}

func (s *State) Copy(other *State) {
	if len(s.S) != len(other.S) {
		s.S = make([]float64, len(other.S))
	}
	copy(s.S, other.S)
	s.BoardSize = other.BoardSize
	s.MaxPieceSize = other.MaxPieceSize
	s.NumPiecesPerSize = other.NumPiecesPerSize
	s.LenPiece = other.LenPiece
	s.CanMove = other.CanMove
	s.lenTower = other.lenTower
	s.bitCurrentPlayer = other.bitCurrentPlayer
}

func (s *State) Player() int {
	return int(s.S[s.bitCurrentPlayer])
}

func (s *State) Win() [numPlayers]bool {
	var win [numPlayers]bool
	for player := 0; player < numPlayers; player++ {
		winningPatterns := calcPatterns(s.BoardSize)
		for _, ptn := range winningPatterns {
			win[player] = s.winByPattern(player, ptn)
			if win[player] {
				break
			}
		}
	}
	return win
}

func (s *State) Next() []Action {
	nexts := make([]Action, 0)
	if s.CanMove {
		nexts = append(nexts, s.nextMovePieces()...)
	}
	nexts = append(nexts, s.nextAddPiece()...)

	nextPlayer := (s.Player() + 1) % numPlayers
	for i := 0; i < len(nexts); i++ {
		nexts[i].State.S[s.bitCurrentPlayer] = float64(nextPlayer)
	}

	return nexts
}

func (s *State) Tower(pos int) Tower {
	return s.S[pos*s.lenTower : (pos+1)*s.lenTower]
}

func (s *State) String() string {
	rowStrs := make([]string, 0, s.BoardSize)
	for y := 0; y < s.BoardSize; y++ {
		towerStrs := make([]string, 0, s.BoardSize)
		for x := 0; x < s.BoardSize; x++ {
			tw := s.Tower(y*s.BoardSize + x)
			towerStrs = append(towerStrs, towerString(s, tw))
		}
		rstr := strings.Join(towerStrs, "|")
		rowStrs = append(rowStrs, rstr)
	}
	stateStr := strings.Join(rowStrs, "\n--------------------\n")
	return stateStr
}

type Action struct {
	State  *State
	IsMove bool
	Src    int
	Dst    int
}

type Piece []float64

func setPiece(pc []float64, player, size int) {
	pc[bitPieceIsPlayed] = 1
	pc[bitPiecePlayer] = float64(player)

	pc[bitPiecePlayer+2] = float64(size % 2)
	quotient := size / 2
	pc[bitPiecePlayer+1] = float64(quotient % 2)
}

func (pc Piece) Size() int {
	return int(pc[bitPiecePlayer+1])*2 + int(pc[bitPiecePlayer+2])
}

func (pc Piece) Player() int {
	if pc[bitPieceIsPlayed] == 0 {
		return PlayerEmpty
	}
	return int(pc[bitPiecePlayer])
}

type Tower []float64

func (s *State) top(tw Tower) (int, Piece) {
	emptyLevel := s.MaxPieceSize
	for i := 0; i < s.MaxPieceSize; i++ {
		pc := Piece(tw[i*s.LenPiece : (i+1)*s.LenPiece])
		if pc.Player() == PlayerEmpty {
			emptyLevel = i
			break
		}
	}

	level := emptyLevel - 1
	if level < 0 {
		return levelEmpty, Piece{}
	}
	return level, Piece(tw[level*s.LenPiece : (level+1)*s.LenPiece])
}

func (s *State) nextMovePiece(pos int) []Action {
	pieceLevel, piece := s.top(s.Tower(pos))
	if pieceLevel == levelEmpty || piece.Player() != s.Player() {
		return nil
	}

	nexts := make([]Action, 0)
	for i := 0; i < s.BoardSize*s.BoardSize; i++ {
		if i == pos {
			continue
		}
		dstLevel, dst := s.top(s.Tower(i))
		if dstLevel != levelEmpty && piece.Size() <= dst.Size() {
			continue
		}

		var act Action
		act.State = NewState(s.BoardSize, s.MaxPieceSize, s.NumPiecesPerSize)
		act.State.Copy(s)
		dstIdx := i*s.lenTower + (dstLevel+1)*s.LenPiece
		for j := 0; j < len(piece); j++ {
			act.State.S[dstIdx+j] = piece[j]
		}
		srcIdx := pos*s.lenTower + pieceLevel*s.LenPiece
		for j := 0; j < len(piece); j++ {
			act.State.S[srcIdx+j] = 0
		}
		act.IsMove = true
		act.Src = pos
		act.Dst = i
		nexts = append(nexts, act)
	}
	return nexts
}

func (s *State) nextMovePieces() []Action {
	nexts := make([]Action, 0)
	for i := 0; i < s.BoardSize*s.BoardSize; i++ {
		nexts = append(nexts, s.nextMovePiece(i)...)
	}
	return nexts
}

func (s *State) deployed() []int {
	counts := make([]int, s.MaxPieceSize)
	for size := 0; size < s.MaxPieceSize; size++ {
		for pos := 0; pos < s.BoardSize*s.BoardSize; pos++ {
			tw := s.S[pos*s.lenTower : (pos+1)*s.lenTower]
			for level := 0; level < s.MaxPieceSize; level++ {
				piece := Piece(tw[level*s.LenPiece : (level+1)*s.LenPiece])
				if piece.Player() != s.Player() {
					continue
				}
				if piece.Size() == size {
					counts[size]++
				}
			}
		}
	}
	return counts
}

func (s *State) nextAddPiece() []Action {
	nexts := make([]Action, 0)

	deployed := s.deployed()
	for size := 0; size < s.MaxPieceSize; size++ {
		if s.NumPiecesPerSize-deployed[size] == 0 {
			continue
		}

		for i := 0; i < s.BoardSize*s.BoardSize; i++ {
			dstLevel, dst := s.top(s.Tower(i))
			if dstLevel != levelEmpty && size <= dst.Size() {
				continue
			}

			var act Action
			act.State = NewState(s.BoardSize, s.MaxPieceSize, s.NumPiecesPerSize)
			act.State.Copy(s)
			dstIdx := i*s.lenTower + (dstLevel+1)*s.LenPiece
			setPiece(act.State.S[dstIdx:dstIdx+s.LenPiece], s.Player(), size)
			act.IsMove = false
			act.Src = size
			act.Dst = i
			nexts = append(nexts, act)
		}
	}

	return nexts
}

func (s *State) winByPattern(player int, pattern []int) bool {
	for i := 0; i < len(pattern); i++ {
		level, piece := s.top(s.Tower(pattern[i]))
		if level == levelEmpty || piece.Player() != player {
			return false
		}
	}
	return true
}

func calcPatterns(boardSize int) [][]int {
	switch boardSize {
	case 3:
		return [][]int{
			{0, 1, 2},
			{3, 4, 5},
			{6, 7, 8},
			{0, 3, 6},
			{1, 4, 7},
			{2, 5, 8},
			{0, 4, 8},
			{2, 4, 6},
		}
	default:
		log.Fatalf("%d", boardSize)
		return nil
	}
}

func towerString(s *State, tw Tower) string {
	pieceStrs := make([]string, 0, s.MaxPieceSize)
	for level := 0; level < s.MaxPieceSize; level++ {
		piece := Piece(tw[level*s.LenPiece : (level+1)*s.LenPiece])
		if piece.Player() == PlayerEmpty {
			pieceStrs = append(pieceStrs, "  ")
			continue
		}

		pstr := ""
		if piece.Player() == 0 {
			pstr += "r"
		} else {
			pstr += "b"
		}
		pstr += fmt.Sprintf("%d", piece.Size())
		pieceStrs = append(pieceStrs, pstr)
	}
	return strings.Join(pieceStrs, "")
}
