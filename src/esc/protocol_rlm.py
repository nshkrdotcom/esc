"""Enforce public submission shape inside native RLM history, without oracle feedback."""
import dspy
from dspy.primitives.code_interpreter import FinalOutput


class ProtocolRLM(dspy.RLM):
    def __init__(self, signature, *, transition_protocol, **kwargs):
        super().__init__(signature, **kwargs)
        self.transition_protocol = transition_protocol

    def _process_execution_result(self, pred, code, result, history, output_field_names):
        if isinstance(result, FinalOutput):
            outputs, error = self._process_final_output(result, output_field_names)
            protocol = self.transition_protocol
            if not error and outputs.get('answer') is not None:
                incomplete = len(protocol.events) != len(protocol.steps)
                rejected = protocol.stopped or bool(protocol.violations)
                inconsistent = outputs['answer'] != protocol.current
                if incomplete or rejected or inconsistent:
                    message = (
                        'Public protocol: answer requires every transition receipt in order. '
                        'Call emit for each step and use its RETURNED value as the next subject. '
                        'Do not reuse precomputed downstream values after a changed receipt. '
                        'If a receipt stopped the chain, or you cannot complete it, submit null. '
                        'No additional iterations or tokens are granted.'
                    )
                    protocol.submission_rejections.append(dict(answer=outputs['answer'],
                        emitted=len(protocol.events), stopped=protocol.stopped))
                    return history.append(reasoning=pred.reasoning,code=code,output='[Error] '+message)
        return super()._process_execution_result(pred, code, result, history, output_field_names)
