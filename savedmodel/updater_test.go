package savedmodel

import (
	"context"
	"os"
	"path"
	"strconv"
	"testing"
	"time"

	"gocloud.dev/blob"
	_ "gocloud.dev/blob/fileblob" // support inmem
)

func getRepoRoot() string {
	return path.Join(getTestPath(), "../..")
}

func TestUpdater(t *testing.T) {
	ctx := context.Background()

	bucket, err := blob.OpenBucket(ctx, "file://"+getRepoRoot())
	if err != nil {
		t.Fatal(err)
	}

	updateChan := make(chan Update)
	errChan := make(chan error)

	updater := NewUpdater(UpdaterConfig{
		Bucket:        bucket,
		Interval:      time.Second * 5,
		ModelName:     "wide_deep",
		ModelsPath:    "testdata/models",
		UpdateChannel: updateChan,
		ErrorChannel:  errChan,
	})

	go func() {
		if err := updater.Start(ctx); err != nil {
			errChan <- err
		}
	}()
	defer updater.Close(ctx)

	select {
	case upd := <-updateChan:
		if upd.ModelName != "wide_deep" {
			t.Error("wrong model name")
		}
		if upd.Version != 1527087570 {
			t.Error("wrong model version")
		}

		if _, err := os.Stat(path.Join(upd.ModelsDir, upd.ModelName, strconv.Itoa(upd.Version), "saved_model.pb")); err != nil {
			t.Error(err)
		}

		if _, err := os.Stat(path.Join(upd.ModelsDir, upd.ModelName, strconv.Itoa(upd.Version), "variables", "variables.index")); err != nil {
			t.Error(err)
		}

		upd.Ack()
	case err := <-errChan:
		t.Error(err)
	}
}

func TestUpdaterWithLatestVersionFile(t *testing.T) {
	ctx := context.Background()

	bucket, err := blob.OpenBucket(ctx, "file://"+getRepoRoot())
	if err != nil {
		t.Fatal(err)
	}

	updateChan := make(chan Update)
	errChan := make(chan error)

	updater := NewUpdater(UpdaterConfig{
		Bucket:            bucket,
		Interval:          time.Second * 5,
		ModelName:         "wide_deep",
		ModelsPath:        "testdata/models",
		LatestVersionFile: "LATEST",
		UpdateChannel:     updateChan,
		ErrorChannel:      errChan,
	})

	go func() {
		if err := updater.Start(ctx); err != nil {
			errChan <- err
		}
	}()
	defer updater.Close(ctx)

	select {
	case upd := <-updateChan:
		if upd.ModelName != "wide_deep" {
			t.Error("wrong model name")
		}
		if upd.Version != 1527087570 {
			t.Error("wrong model version")
		}

		if _, err := os.Stat(path.Join(upd.ModelsDir, upd.ModelName, strconv.Itoa(upd.Version), "saved_model.pb")); err != nil {
			t.Error(err)
		}

		if _, err := os.Stat(path.Join(upd.ModelsDir, upd.ModelName, strconv.Itoa(upd.Version), "variables", "variables.index")); err != nil {
			t.Error(err)
		}

		upd.Ack()
	case err := <-errChan:
		t.Error(err)
	}
}

func TestUpdaterOnError(t *testing.T) {
	ctx := context.Background()

	bucket, err := blob.OpenBucket(ctx, "file://"+getRepoRoot())
	if err != nil {
		t.Fatal(err)
	}

	updateChan := make(chan Update)
	errChan := make(chan error)

	updater := NewUpdater(UpdaterConfig{
		Bucket:        bucket,
		Interval:      time.Second * 5,
		ModelName:     "model_doesnt_exist",
		ModelsPath:    "testdata/models",
		UpdateChannel: updateChan,
		ErrorChannel:  errChan,
	})

	err = updater.Start(ctx)
	if err != ErrNoModelFound {
		t.Error("Should have returned an error")
	}
}
