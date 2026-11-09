from contracts import SearchBudget


def unified_gpu_budget(budget: SearchBudget) -> SearchBudget:
    """校准、短程验证及正式训练共同使用总GPU额度。"""
    return budget.model_copy(update={
        'max_profiling_gpu_hours': budget.max_total_gpu_hours,
    })
