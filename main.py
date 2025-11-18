from cerebrum.application.bootstrap import build_container
from cerebrum.presentation.cli.session import CliSession


def main() -> None:
	with build_container() as container:
		CliSession(container.service, container.language_model, container.speech_to_text).run_session()


if __name__ == "__main__":
	main()
