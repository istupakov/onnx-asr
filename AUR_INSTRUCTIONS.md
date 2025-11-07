# Packaging onnx-asr for the Arch User Repository (AUR)

This guide provides step-by-step instructions on how to use the `PKGBUILD` file to create an Arch Linux package for `onnx-asr` and submit it to the AUR.

## Prerequisites

Before you begin, make sure you have the following packages installed on your Arch Linux system:

- `base-devel`: This package group includes essential tools for building packages, including `makepkg`.
- `git`: This is required for interacting with the AUR.

You can install these with the following command:

```bash
sudo pacman -S --needed base-devel git
```

## Building and Installing the Package Locally

1.  **Create a directory and download the `PKGBUILD`:**

    ```bash
    mkdir onnx-asr-aur
    cd onnx-asr-aur
    # Move the PKGBUILD file into this directory
    ```

2.  **Generate the checksums:**

    The `PKGBUILD` file has a placeholder for the `sha256sums`. You can automatically generate the correct checksums using `makepkg`:

    ```bash
    makepkg -g
    ```

    This will download the source code and print the checksums. Copy and paste these into the `sha256sums` array in your `PKGBUILD` file.

3.  **Build and install the package:**

    Now you can build and install the package using `makepkg`:

    ```bash
    makepkg -si
    ```

    - `-s` will automatically install the necessary dependencies from the official repositories.
    - `-i` will install the package on your system after it has been successfully built.

## Submitting the Package to the AUR

1.  **Create an AUR account:**

    If you don't already have one, create an account on the [AUR website](https://aur.archlinux.org/).

2.  **Clone the AUR repository:**

    You will need to clone the AUR repository for your package. If the package doesn't exist yet, you'll need to create it first on the AUR website. Once you have, you can clone it:

    ```bash
    git clone ssh://aur@aur.archlinux.org/python-onnx-asr.git
    ```

3.  **Add the `PKGBUILD` and `.SRCINFO`:**

    Copy your `PKGBUILD` file into the cloned repository. Then, you need to generate a `.SRCINFO` file, which contains metadata about the package:

    ```bash
    cd python-onnx-asr
    makepkg --printsrcinfo > .SRCINFO
    ```

4.  **Commit and push the changes:**

    Now you can add the files to the Git repository, commit them, and push them to the AUR:

    ```bash
    git add PKGBUILD .SRCINFO
    git commit -m "Initial import"
    git push
    ```

## Maintaining the Package

When a new version of `onnx-asr` is released, you will need to update the `PKGBUILD` file. This involves:

1.  Updating the `pkgver` variable.
2.  Updating the `sha256sums` by running `makepkg -g`.
3.  Committing and pushing the changes to the AUR.
