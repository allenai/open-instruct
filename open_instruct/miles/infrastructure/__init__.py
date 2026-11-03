"""Share small operational helpers across launch, training and evaluation.
Atomic JSON writes keep receipts readable during updates, while timing and timeout
helpers make slow infrastructure operations visible without importing GPU code.
"""
