# Maintainer: Your Name <your.email@example.com>

pkgname=python-onnx-asr
_name=onnx-asr
pkgver=0.4.5 # This should be updated to the latest version
pkgrel=1
pkgdesc="Automatic Speech Recognition in Python using ONNX models"
arch=('any')
url="https://github.com/istupakov/onnx-asr"
license=('MIT')
depends=('python' 'python-numpy' 'python-onnxruntime')
makedepends=('python-pdm' 'python-build' 'python-installer' 'python-wheel')
source=("${_name}-${pkgver}.tar.gz::https://github.com/istupakov/${_name}/archive/refs/tags/v${pkgver}.tar.gz")
sha256sums=('SKIP') # This should be filled in later with makepkg -g

build() {
    cd "${_name}-${pkgver}"
    # The project uses pdm but we can use a standard PEP 517 build
    python -m build --wheel --no-isolation
}

package() {
    cd "${_name}-${pkgver}"
    python -m installer --destdir="$pkgdir/" dist/*.whl
}
