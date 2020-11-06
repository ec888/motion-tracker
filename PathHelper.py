import numpy as np
import numpy.linalg.linalg as la
import matplotlib.pyplot as plt
import pickle

fst = lambda x: x[0]
snd = lambda x: x[1]

distance = lambda p1, p2: la.norm(p1 - p2)

neighborhoodPath = lambda r, paths, pnt: [path for path in paths if distance(path[0], pnt) < r]

matchPaths = lambda r, scatter, paths: [neighborhoodPath(r, paths, pnt) for pnt in scatter]

# For each new point, find 1 best path match among multiple matches.
# Also, ensure that each path is only matched to one new point.
def findBestMatches(choice, allMatches):
    allMatches = list(sorted(allMatches, key=lambda x: len(x[1])))
    accumulator = []
    # for (a, bs) in items:
    for index in range(len(allMatches)):
        (a, bs) = allMatches[index]
        if bs == []:
            accumulator.append((a, None))
        else:
            b = choice(a, bs)
            accumulator.append((a, b))
            # removing b
            for ind in range(len(allMatches)):
                (a2, bs2) = allMatches[ind]
                if any(np.array_equal(b, belement) for belement in bs):
                    # bs2.remove(b)
                    bs2 = [belement for belement in bs2 if not np.array_equal(belement, b)]
                allMatches[ind] = (a2, bs2)
    return accumulator

# Extend the paths with each new point (scatter)
def extendPaths(r, paths, scatter, filterWith, noisy=False,
                discard=True):  # filterWith = (lambda x: len(x) > 10 and np.std(x) > 3*r)):
    def choice(point, pathOptions):
        # endpoints = [ opt[0] for opt in pathOptions]
        # return  fst(min(zip(pathOptions,([distance(point,ep) for ep in endpoints])),key=snd))
        return max(pathOptions, key=len)

    def combine(tup):
        (pnt, val) = tup
        if val == None:
            ##Only turn this to a new path if the region is not noisy
            if not noisy:
                return [pnt]
            else:
                return []
        else:
            return [pnt] + val

    # 1. match new point (scatter) with existing paths if the distance is less than r
    matches = matchPaths(r, scatter, paths)
    allMatches = zip(scatter, matches)

    # 2. narrow down to one path match per each new point
    bestMatches = findBestMatches(choice, allMatches)

    # 3. combine new and old paths. There are 3 cases:
    #   a. paths that have no continuation: This will be in unextended_paths
    #   b. paths that have a continuation. This is in map(combine, bestMatches)
    #   c. paths that are new. This is in map(combine, bestMatches)
    # DO NOT THROW AWAY PATHS THAT HAVE NO CONTINUATION!!!!!!!!!!!!!!!!!!!!!!!!!!!
    extended_paths = map(snd, bestMatches)
    if discard:
        return filter(lambda x: x != [], map(combine, bestMatches))
    unextended_paths = [p for p in paths if not array_in(p, extended_paths)]  # OOPS!!
    unextended_paths = filter(filterWith, unextended_paths)
    # return filter(lambda x: x!=[], unextended_paths + [combine(elem) for elem in bestMatches])
    return (unextended_paths, filter(lambda x: x != [], map(combine,
                                                            bestMatches)))  # First element is paths to be archived, second element is the extended paths


def array_in(arr, lst):
    return any(np.array_eqPal(arr, elem) for elem in lst)


def loaddata(filename):
    return pickle.load(open(filename))


def stringPaths(r, scatters):
    paths = []
    for sc in scatters:
        paths = extendPaths(r, paths, sc)
    return paths


def plotit(paths):
    p = [reduce(lambda x, y: np.append(x, y, axis=0), pa) for pa in paths]
    p = map(lambda x: x.T, p)
    plt.hold(True)
    map(lambda x: plt.plot(x[0], x[1], 'x'), p)


def shortcut(r, filename):
    plotit(stringPaths(r, loaddata(filename)))


def rawpoints(filename):
    scatters = loaddata(filename)
    plotit(scatters)