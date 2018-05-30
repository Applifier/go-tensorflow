package savedmodel

import (
	"errors"
	"io"
	"io/ioutil"

	protobuf "github.com/Applifier/go-tensorflow/types/tensorflow/core/protobuf"
)

// ErrSignatureNotFound error returned then given signature was not found from SavedModel
var ErrSignatureNotFound = errors.New("no signature with given name found")

// GetSignatureDefFromReader returns SignatureDef from a given io.Reader
func GetSignatureDefFromReader(tags []string, signature string, r io.Reader) (*protobuf.SignatureDef, error) {
	savedModel := &protobuf.SavedModel{}

	b, err := ioutil.ReadAll(r)
	if err != nil {
		return nil, err
	}

	if err := savedModel.Unmarshal(b); err != nil {
		return nil, err
	}

	metagraphs := savedModel.GetMetaGraphs()
	for _, mgraph := range metagraphs {
		info := mgraph.GetMetaInfoDef()
		mGraphTags := info.GetTags()

		if !compareStringSlice(mGraphTags, tags) {
			continue
		}

		def, ok := mgraph.SignatureDef[signature]
		if ok {
			return def, nil
		}
	}

	return nil, ErrSignatureNotFound
}

func compareStringSlice(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}

	for _, aitem := range a {
		found := false
		for _, bitem := range b {
			if aitem == bitem {
				found = true
				break
			}
		}
		if !found {
			return false
		}
	}

	return true
}
