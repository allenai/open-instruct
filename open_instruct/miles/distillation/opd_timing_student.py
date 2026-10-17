"""Time the native student callable, after admission and before teacher scoring."""

from miles.rollout.generate_hub import single_turn

from open_instruct.miles.distillation import opd_timing


async def generate(input):
    sample = input.sample
    opd_timing.student_admitted(sample, input.evaluation)
    with opd_timing.stage("student_request", evaluation=input.evaluation, **opd_timing.identity(sample)) as row:
        result = sample
        try:
            output = await single_turn.generate(input)
            result = output.samples
            return output
        finally:
            row.update(response_tokens=result.response_length, status=result.status.value)
