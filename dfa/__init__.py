"""
DNA fibers analysis package.
"""
__all__ = [
    'modeling', 'detection', 'extraction', 'analysis', 'simulation',
    'comparison']

# gold number (continuous fraction approximation, current iteration is 7)
# def dfa_version(n):
#     def _approx_gold_rec(e, n):
#         if n == 0:
#             return 1 + e
#         else:
#             return _approx_gold_rec(1/(1 + e), n - 1)
#     return str(_approx_gold_rec(1, n))
__version__ = '1.6176470588235294'
