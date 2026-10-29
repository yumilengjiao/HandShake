from shared.contracts import ActionResult, Capability


class DryRunBackend:
    """Records proposed commands, never reports operating-system injection."""
    def __init__(self, capabilities=(Capability.PRESENTATION,)):
        self.available = set(capabilities)
        self.commands = []

    def capabilities(self):
        return set(self.available)

    def execute(self, command, now_ns):
        self.commands.append(command)
        return ActionResult(command_id=command.command_id, accepted=True, injection_status="not_attempted",
                            completed_at_ns=now_ns, acknowledgement_level="none", timebase=command.timebase)

    def release_all(self):
        return True
