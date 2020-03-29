import numpy
from scipy.ndimage import correlate1d, convolve1d
import support
import cv2

def _gaussian_kernel1d(sigma, order, radius):
    if order < 0:
        raise ValueError('order must be non-negative')
    exponent_range = numpy.arange(order + 1)
    sigma2 = sigma * sigma
    x = numpy.arange(-radius, radius + 1)
    phi_x = numpy.exp(-0.5 / sigma2 * x ** 2)
    phi_x = phi_x / phi_x.sum()

    if order == 0:
        return phi_x
    else:
        q = numpy.zeros(order + 1)
        q[0] = 1
        D = numpy.diag(exponent_range[1:], 1)  # D @ q(x) = q'(x)
        P = numpy.diag(numpy.ones(order) / -sigma2, -1)  # P @ q(x) = q(x) * p'(x)
        Q_deriv = D + P
        for _ in range(order):
            q = Q_deriv.dot(q)
        q = (x[:, None] ** exponent_range).dot(q)
        return q * phi_x


def gaussian_filter1d(input, sigma, axis=-1, order=0, output=None,
                      mode="reflect", cval=0.0, truncate=4.0):
    sd = float(sigma)
    lw = int(truncate * sd + 0.5)
    weights = _gaussian_kernel1d(sigma, order, lw)[::-1]
    return correlate1d(input, weights, axis, output, mode, cval, 0)


def gaussian_filter(input, sigma, order=0, output=None,
                    mode="reflect", cval=0.0, truncate=4.0):
    input = numpy.asarray(input)
    output = support._get_output(output, input)
    orders = support._normalize_sequence(order, input.ndim)
    sigmas = support._normalize_sequence(sigma, input.ndim)
    modes = support._normalize_sequence(mode, input.ndim)
    axes = list(range(input.ndim))
    axes = [(axes[ii], sigmas[ii], orders[ii], modes[ii])
            for ii in range(len(axes)) if sigmas[ii] > 1e-15]
    if len(axes) > 0:
        for axis, sigma, order, mode in axes:
            gaussian_filter1d(input, sigma, axis, order, output,
                              mode, cval, truncate)
            input = output
    else:
        output[...] = input[...]
    return output


def threshold_adaptive(image, block_size, method='gaussian', offset=0,
                       mode='reflect', param=None, cval=0):
    if block_size % 2 == 0:
        raise ValueError("The kwarg ``block_size`` must be odd! Given "
                         "``block_size`` {0} is even.".format(block_size))
    thresh_image = numpy.zeros(image.shape, 'double')
    if param is None:
        sigma = (block_size - 1) / 6.0
    else:
        sigma = param
    gaussian_filter(image, sigma, output=thresh_image, mode=mode,
                        cval=cval)

    return thresh_image - offset


def _preprocess_input(image, selem=None, out=None, mask=None, out_dtype=None,
                      pixel_size=1):
    input_dtype = image.dtype
    if (input_dtype in (bool, numpy.bool, numpy.bool_)
            or out_dtype in (bool, numpy.bool, numpy.bool_)):
        raise ValueError('dtype cannot be bool.')
    if input_dtype not in (numpy.uint8, numpy.uint16):
        image = support.img_as_ubyte(image)

    selem = numpy.ascontiguousarray(support.img_as_ubyte(selem > 0))
    image = numpy.ascontiguousarray(image)

    if out is None:
        if out_dtype is None:
            out_dtype = image.dtype
        out = numpy.empty(image.shape + (pixel_size,), dtype=out_dtype)
    else:
        if len(out.shape) == 2:
            out = out.reshape(out.shape + (pixel_size,))
    if image.dtype in (numpy.uint8, numpy.int8):
        n_bins = 256
    else:
        n_bins = int(max(3, image.max())) + 1
    return image, selem, out, mask, n_bins


def _apply_scalar_per_pixel(func, image, selem, out, mask, shift_x, shift_y,
                            out_dtype=None):
    image, selem, out, mask, n_bins = _preprocess_input(image, selem,
                                                        out, mask,
                                                        out_dtype)
    func(image, selem, shift_x=shift_x, shift_y=shift_y, mask=mask,
         out=out, n_bins=n_bins)

    return numpy.squeeze(out, axis=-1)


def local_mean(image, block_size=3, offset=0):
    '''return cv2.medianBlur(image, 3)'''
    thresh_image = numpy.zeros(image.shape, 'double')
    mask = 1. / block_size * numpy.ones((block_size,))
    convolve1d(image, mask, axis=0, output=thresh_image, mode='reflect',
                   cval=0)
    convolve1d(thresh_image, mask, axis=1, output=thresh_image,
                   mode='reflect', cval=0)
    return thresh_image - offset


def local_medium(image):
    kernel = numpy.ones((3, 3), numpy.float32) / 9
    return cv2.filter2D(image, -1, kernel)
