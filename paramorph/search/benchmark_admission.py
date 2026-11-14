import math


def native_validation_passed(selection):
    validation = selection.get('validation') or {}
    identifier = selection['id']
    if validation.get('id') != identifier or not selection.get('validation_trace'):
        return False
    review = validation.get('review')
    if review is None:
        return True
    evaluation = selection.get('calibration_selection') or {}
    approved = {review['id'], *(item['id'] for item in review['deferred'] if item['eligible_for_calibration'] is True)}
    return (identifier in approved and evaluation.get('candidate_id') == identifier
        and evaluation.get('micro_batch') == selection['strategy']['micro_batch']
        and bool(selection.get('calibration_selection_trace')))


def formal_rejection(*, validated, complete_graph, predicted_rate, leader, memory_admitted, peaks, capacities):
    if not validated:
        return 'VALIDATION_REQUIRED'
    if not complete_graph or type(predicted_rate) not in (float, int) or not math.isfinite(predicted_rate) or predicted_rate <= 0:
        return 'PROFILE_REQUIRED_BEFORE_BENCHMARK'
    if leader is not None and predicted_rate <= leader:
        return 'PREDICTED_NOT_FASTER_THAN_MEASURED_LEADER'
    if (not memory_admitted or not capacities or set(peaks) != set(capacities)
            or any(type(value) not in (float, int) or not math.isfinite(value)
                or not 0 < value < capacities[rank] for rank, value in peaks.items())):
        return 'MEMORY_NOT_ADMITTED'
    return None
