package savedmodel

import (
	"context"
	"errors"
	"io"
	"io/ioutil"
	"os"
	"path"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"gocloud.dev/blob"
)

// ErrNotAcknowledged returned if recipient calls Nack on the update
var ErrNotAcknowledged = errors.New("not acked")

// ErrNoModelFound returned if no models found
var ErrNoModelFound = errors.New("no models")

// Update struct contains information of the location for a new Tensorflow model
type Update struct {
	ModelsDir string
	ModelName string
	Version   int

	done func(err error)
}

// Ack acknowledges model update as being processed correctly
func (u *Update) Ack() {
	u.done(nil)
}

// Nack informs that model update could not be processed and it should be retried until a new model is available
func (u *Update) Nack() {
	u.done(errors.New(""))
}

// UpdaterConfig struct containing model updater config
type UpdaterConfig struct {
	Interval      time.Duration
	ModelName     string
	ModelsPath    string
	Bucket        *blob.Bucket
	UpdateChannel chan Update
	ErrorChannel  chan error
	// AllowedUpdateFailures numbers of sequential update failures allowed (-1 = infinite)
	AllowUpdateFailures int
	LatestVersionFile   string
}

// Updater poll for models updates in a given blob.Bucket and path
type Updater struct {
	config UpdaterConfig

	failures      int
	closeChan     chan struct{}
	latestVersion int
}

// NewUpdater returns a new updated for given
func NewUpdater(config UpdaterConfig) *Updater {
	u := &Updater{
		config:    config,
		closeChan: make(chan struct{}),
	}

	return u
}

func (u *Updater) checkForUpdate(ctx context.Context) error {
	var ver int
	var err error

	if u.config.LatestVersionFile == "" {
		ver, err = u.getLatestVersionForDirs(ctx)
	} else {
		ver, err = u.getLatestVersionForFile(ctx)
	}

	if err != nil {
		return err
	}

	tempDir, err := u.downloadModelToTemp(ctx, ver)
	if err != nil {
		return err
	}

	upd := Update{
		ModelsDir: tempDir,
		ModelName: u.config.ModelName,
		Version:   ver,
	}

	// Remove temp dir once pushUpdate returns (update has been processed)
	defer os.RemoveAll(tempDir)

	return u.pushUpdate(upd)
}

func (u *Updater) downloadModelToTemp(ctx context.Context, version int) (string, error) {
	tempDir, err := ioutil.TempDir("", "tf_models")
	if err != nil {
		return "", err
	}

	files, err := u.listModelFiles(ctx, version)
	if err != nil {
		return "", err
	}

	for _, file := range files {
		reader, err := u.getModelFile(ctx, version, file)
		if err != nil {
			return "", err
		}
		defer reader.Close()

		if err := writeTo(reader, path.Join(tempDir, u.config.ModelName, strconv.Itoa(version), file)); err != nil {
			return "", err
		}
	}

	return tempDir, nil
}

func (u *Updater) getLatestVersionForFile(ctx context.Context) (int, error) {
	r, err := u.getFile(ctx, u.config.LatestVersionFile)
	if err != nil {
		return 0, err
	}

	b, err := ioutil.ReadAll(r)
	if err != nil {
		return 0, err
	}

	return strconv.Atoi(strings.TrimSpace(string(b)))
}

func (u *Updater) getFile(ctx context.Context, file string) (io.ReadCloser, error) {
	key := filepath.Join(u.config.ModelsPath, u.config.ModelName, file)
	return u.config.Bucket.NewReader(ctx, key, nil)
}

func (u *Updater) getModelFile(ctx context.Context, version int, file string) (io.ReadCloser, error) {
	key := filepath.Join(u.config.ModelsPath, u.config.ModelName, strconv.Itoa(version), file)
	return u.config.Bucket.NewReader(ctx, key, nil)
}

func (u *Updater) listModelFiles(ctx context.Context, version int) (files []string, err error) {
	root := filepath.Join(u.config.ModelsPath, u.config.ModelName, strconv.Itoa(version))
	if root != "" && !strings.HasSuffix(root, "/") {
		root = root + "/"
	}

	var walk func(path string) error
	walk = func(path string) error {
		iter := u.config.Bucket.List(&blob.ListOptions{
			Prefix:    path,
			Delimiter: "/",
		})

		for {
			obj, err := iter.Next(ctx)
			if err != nil {
				if err == io.EOF {
					return nil
				}
				return err
			}

			if obj.IsDir {
				if err := walk(obj.Key); err != nil {
					return err
				}
				continue
			}

			filename := obj.Key[len(root):]
			files = append(files, filename)
		}
	}

	if err = walk(root); err != nil {
		return nil, err
	}

	return files, nil
}

func (u *Updater) getLatestVersionForDirs(ctx context.Context) (int, error) {
	path := path.Join(u.config.ModelsPath, u.config.ModelName)
	if path != "" && !strings.HasSuffix(path, "/") {
		path = path + "/"
	}

	iter := u.config.Bucket.List(&blob.ListOptions{
		Prefix:    path,
		Delimiter: "/",
	})

	latestVersion := 0

	for {
		obj, err := iter.Next(ctx)
		if err != nil {
			if err == io.EOF {
				if latestVersion == 0 {
					return 0, ErrNoModelFound
				}
				return latestVersion, nil
			}
			return 0, err
		}

		if obj.IsDir {
			parts := strings.Split(strings.TrimSuffix(obj.Key, "/"), "/")
			versionStr := parts[len(parts)-1]

			if ver, err := strconv.Atoi(versionStr); err == nil {
				if latestVersion < ver {
					latestVersion = ver
				}
			}

		}
	}
}

func (u *Updater) pushUpdate(upd Update) error {

	errChan := make(chan error, 1)

	select {
	case <-u.closeChan:
	default:
		upd.done = func(err error) {
			defer func() {
				errChan <- err
			}()

			if err != nil {
				if u.config.ErrorChannel != nil {
					u.config.ErrorChannel <- err
				}
				return
			}
		}
		u.config.UpdateChannel <- upd
	}

	if err := <-errChan; err != nil {
		return err
	}

	u.latestVersion = upd.Version

	return nil
}

// Start starts polling for updates and block until an error occures
func (u *Updater) Start(ctx context.Context) error {
	ticker := time.NewTicker(u.config.Interval)
	defer ticker.Stop()

	for {
		err := u.checkForUpdate(ctx)
		if err != nil {
			u.failures++

			if u.config.AllowUpdateFailures >= 0 && u.failures > u.config.AllowUpdateFailures {
				return err
			}

			if u.config.ErrorChannel != nil {
				u.config.ErrorChannel <- err
			}
			return nil
		}

		u.failures = 0

		select {
		case <-u.closeChan:
			return nil
		case <-ticker.C:
		}
	}
}

// Close closes updater and causes Start to return
func (u *Updater) Close(ctx context.Context) error {
	close(u.closeChan)
	return nil
}

func writeTo(r io.Reader, target string) error {
	if err := os.MkdirAll(path.Dir(target), 0700); err != nil {
		return err
	}

	f, err := os.Create(target)
	if err != nil {
		return err
	}
	defer f.Close()

	if _, err := io.Copy(f, r); err != nil {
		return err
	}

	return nil
}
