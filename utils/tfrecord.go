package utils

import (
	"bytes"
	"encoding/binary"
	"hash/crc32"
	"io"

	"github.com/pkg/errors"
)

const (
	maskDelta = 0xa282ead8

	headerSize = 12
	footerSize = 4
)

var crc32c = crc32.MakeTable(crc32.Castagnoli)

// ErrInvalidChecksum is returned if header or footer checksum is invalid
var ErrInvalidChecksum = errors.New("invalid crc")

// TFRecordReader reads tf records from a given io.Reader
type TFRecordReader struct {
	reader io.Reader
	bytes  []byte
}

// NewTFRecordReader returns a new TFRecordReader for a given io.Reader
func NewTFRecordReader(reader io.Reader) *TFRecordReader {
	return &TFRecordReader{
		reader: reader,
	}
}

// Next returns the next TFRecord or io.EOF error if the files have been consumed
func (tfr *TFRecordReader) Next() error {
	f := tfr.reader

	buf := bytes.NewBuffer(nil)
	buf.Grow(headerSize * footerSize)

	_, err := io.CopyN(buf, f, headerSize)
	if err != nil {
		return err
	}

	header := buf.Bytes()

	crc := binary.LittleEndian.Uint32(header[8:12])
	if !verifyChecksum(header[0:8], crc) {
		return errors.Wrap(ErrInvalidChecksum, "length")
	}

	length := binary.LittleEndian.Uint64(header[0:8])
	buf.Reset()

	if _, err = io.CopyN(buf, f, int64(length)); err != nil {
		return err
	}

	if _, err = io.CopyN(buf, f, footerSize); err != nil {
		return err
	}

	payload := buf.Bytes()

	footer := payload[length:]
	crc = binary.LittleEndian.Uint32(footer)
	if !verifyChecksum(payload[:length], crc) {
		return errors.Wrap(ErrInvalidChecksum, "payload")
	}

	tfr.bytes = payload[:length]

	return nil
}

// Bytes returns the current bytes from the reader
func (tfr *TFRecordReader) Bytes() []byte {
	return tfr.bytes
}

func verifyChecksum(data []byte, crcMasked uint32) bool {
	rot := crcMasked - maskDelta
	unmaskedCrc := ((rot >> 17) | (rot << 15))

	crc := crc32.Checksum(data, crc32c)

	return crc == unmaskedCrc
}
