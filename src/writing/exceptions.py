class ProjectionUpdateFailed(RuntimeError):
    """Projection 更新失败，用于阻断不完整提交"""
    pass


class SceneCompletionFailed(RuntimeError):
    """场景完成失败，用于整体事务回滚"""
    pass