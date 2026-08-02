"""Bulk 2D geometry kernels exposed to the Python bindings.

The predicates report zero when the floating-point filter cannot certify a
sign.  Python then evaluates that exceptional case exactly with dyadic rationals.
"""

comptime Ptr = UnsafePointer[Float64, AnyOrigin[mut=True]]
comptime IPtr = UnsafePointer[Int, AnyOrigin[mut=True]]


def sign(x: Float64) -> Int:
    if x > 0.0:
        return 1
    if x < 0.0:
        return -1
    return 0


def abs64(x: Float64) -> Float64:
    if x < 0.0:
        return -x
    return x


def orient_filtered(ax: Float64, ay: Float64, bx: Float64, by: Float64, cx: Float64, cy: Float64) -> Int:
    var acx = ax - cx
    var bcx = bx - cx
    var acy = ay - cy
    var bcy = by - cy
    var left = acx * bcy
    var right = acy * bcx
    var det = left - right
    var bound = (abs64(left) + abs64(right)) * 3.3306690738754716e-16
    if abs64(det) > bound:
        return sign(det)
    return 0


def incircle_filtered(ax: Float64, ay: Float64, bx: Float64, by: Float64, cx: Float64, cy: Float64, dx: Float64, dy: Float64) -> Int:
    var adx = ax - dx
    var ady = ay - dy
    var bdx = bx - dx
    var bdy = by - dy
    var cdx = cx - dx
    var cdy = cy - dy
    var abdet = adx * bdy - bdx * ady
    var bcdet = bdx * cdy - cdx * bdy
    var cadet = cdx * ady - adx * cdy
    var alift = adx * adx + ady * ady
    var blift = bdx * bdx + bdy * bdy
    var clift = cdx * cdx + cdy * cdy
    var det = alift * bcdet + blift * cadet + clift * abdet
    var permanent = abs64(bcdet) * alift + abs64(cadet) * blift + abs64(abdet) * clift
    if abs64(det) > permanent * 1.1102230246251565e-15:
        return sign(det)
    return 0


@export("msg_orient2d_batch")
def msg_orient2d_batch(points: Int, n: Int, result: Int) abi("C"):
    if n <= 0:
        return
    var p = Ptr(unsafe_from_address=points)
    var dst = IPtr(unsafe_from_address=result)
    for i in range(n):
        var j = i * 6
        dst.store(i, orient_filtered(p.load(j), p.load(j + 1), p.load(j + 2), p.load(j + 3), p.load(j + 4), p.load(j + 5)))


@export("msg_incircle_batch")
def msg_incircle_batch(points: Int, n: Int, result: Int) abi("C"):
    if n <= 0:
        return
    var p = Ptr(unsafe_from_address=points)
    var dst = IPtr(unsafe_from_address=result)
    for i in range(n):
        var j = i * 8
        dst.store(i, incircle_filtered(p.load(j), p.load(j + 1), p.load(j + 2), p.load(j + 3), p.load(j + 4), p.load(j + 5), p.load(j + 6), p.load(j + 7)))


@export("msg_polygon_area")
def msg_polygon_area(vertices: Int, n: Int) abi("C") -> Float64:
    if n < 3:
        return 0.0
    var p = Ptr(unsafe_from_address=vertices)
    var total = 0.0
    for i in range(n):
        var j = i * 2
        var k = ((i + 1) % n) * 2
        total += p.load(j) * p.load(k + 1) - p.load(j + 1) * p.load(k)
    return total * 0.5


def classify_polygon_point(q: Ptr, p: Ptr, dst: IPtr, n: Int, i: Int):
    var x = q.load(i * 2)
    var y = q.load(i * 2 + 1)
    var winding = 0
    for e in range(n - 1):
        var a = e * 2
        var b = a + 2
        var ax = p.load(a)
        var ay = p.load(a + 1)
        var bx = p.load(b)
        var by = p.load(b + 1)
        var side = orient_filtered(ax, ay, bx, by, x, y)
        # A filtered zero is either a true zero or an uncertified sign.  The
        # Python layer resolves this sentinel with exact dyadic arithmetic.
        if side == 0:
            dst.store(i, 2)
            return
        if ay <= y:
            if by > y and side > 0:
                winding += 1
        elif by <= y and side < 0:
            winding -= 1
    var a = (n - 1) * 2
    var ax = p.load(a)
    var ay = p.load(a + 1)
    var bx = p.load(0)
    var by = p.load(1)
    var side = orient_filtered(ax, ay, bx, by, x, y)
    if side == 0:
        dst.store(i, 2)
        return
    if ay <= y:
        if by > y and side > 0:
            winding += 1
    elif by <= y and side < 0:
        winding -= 1
    if winding == 0:
        dst.store(i, 0)
    else:
        dst.store(i, 1)


@export("msg_points_in_polygon")
def msg_points_in_polygon(points: Int, m: Int, polygon: Int, n: Int, result: Int) abi("C"):
    if m <= 0 or n < 3:
        return
    var q = Ptr(unsafe_from_address=points)
    var p = Ptr(unsafe_from_address=polygon)
    var dst = IPtr(unsafe_from_address=result)
    for i in range(m):
        classify_polygon_point(q, p, dst, n, i)


@export("msg_segment_pairs")
def msg_segment_pairs(segments: Int, n: Int, result: Int) abi("C"):
    if n <= 0:
        return
    var p = Ptr(unsafe_from_address=segments)
    var dst = IPtr(unsafe_from_address=result)
    for i in range(n):
        var j = i * 8
        var a = orient_filtered(p.load(j), p.load(j + 1), p.load(j + 2), p.load(j + 3), p.load(j + 4), p.load(j + 5))
        var b = orient_filtered(p.load(j), p.load(j + 1), p.load(j + 2), p.load(j + 3), p.load(j + 6), p.load(j + 7))
        var c = orient_filtered(p.load(j + 4), p.load(j + 5), p.load(j + 6), p.load(j + 7), p.load(j), p.load(j + 1))
        var d = orient_filtered(p.load(j + 4), p.load(j + 5), p.load(j + 6), p.load(j + 7), p.load(j + 2), p.load(j + 3))
        if a * b < 0 and c * d < 0:
            dst.store(i, 1)
        elif a == 0 or b == 0 or c == 0 or d == 0:
            dst.store(i, 2)
        else:
            dst.store(i, 0)
