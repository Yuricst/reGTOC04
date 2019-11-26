import collections
import functools

# taken from https://wiki.python.org/moin/PythonDecoratorLibrary#Memoize

# memoization allows the object to store previously computed values and keep them on hand.
# this can signficantly improve the speed of an algorithm, at the cost of memory
# for our problem, since we only need to know the positions of object at specific times, it makes sense to store these in memory.
# we may run into a memory space issue, since for 1436 asteroids, and

class memoized(object):
   '''Decorator. Caches a function's return value each time it is called.
   If called later with the same arguments, the cached value is returned
   (not reevaluated).
   '''
   def __init__(self, func):
      self.func = func
      self.cache = {}
   def __call__(self, *args, **kwargs):

      key = (args, frozenset(kwargs.items()))

      if not isinstance(key, collections.Hashable):
         # uncacheable. a list, for instance.
         # better to not cache than blow up.
         return self.func(*args, **kwargs)
      if key in self.cache:
         return self.cache[key]
      else:
         value = self.func(*args, **kwargs)
         self.cache[key] = value
         return value
   def __repr__(self):
      '''Return the function's docstring.'''
      return self.func.__doc__
   def __get__(self, obj, objtype):
      '''Support instance methods.'''
      return functools.partial(self.__call__, obj)

# note that this decorator ignores **kwargs
def memoize(obj):
    cache = obj.cache = {}

    @functools.wraps(obj)
    def memoizer(*args, **kwargs):
        if args not in cache:
            cache[args] = obj(*args, **kwargs)
        return cache[args]
    return memoizer


# example
@memoized
def fibonacci(n):
   "Return the nth fibonacci number."
   if n in (0, 1):
      return n
   return fibonacci(n-1) + fibonacci(n-2)
