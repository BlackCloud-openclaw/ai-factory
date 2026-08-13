# src/writing/snapshot/runtime/remote/gc/errors.py

from ...exceptions import SnapshotRuntimeError


class GarbageCollectionError(SnapshotRuntimeError):
    pass


class GCNotSupportedError(GarbageCollectionError):
    pass


class GCInconsistentError(GarbageCollectionError):
    pass


class LeaseError(GarbageCollectionError):
    pass


class LeaseConflictError(LeaseError):
    pass


class LeaseAcquisitionError(LeaseError):
    pass


class LeaseRenewalError(LeaseError):
    pass


class LeaseReleaseError(LeaseError):
    pass


class DeletionMarkerError(GarbageCollectionError):
    pass


class GracePeriodNotElapsedError(GarbageCollectionError):
    pass


class DeletionFailedError(GarbageCollectionError):
    pass


# B4.8
class MarkerScannerError(GarbageCollectionError):
    pass


class MarkerReconciliationError(GarbageCollectionError):
    pass