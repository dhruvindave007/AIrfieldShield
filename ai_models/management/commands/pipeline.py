# ai_models/management/commands/pipeline.py
from django.core.management.base import BaseCommand
from django.core import management


class Command(BaseCommand):
    help = "Full AI pipeline: Train ensemble + Predict + Refresh alerts"

    def add_arguments(self, parser):
        parser.add_argument(
            "--samples", type=int, default=5000,
            help="Synthetic samples for training"
        )
        parser.add_argument(
            "--seq-len", type=int, default=30,
            help="Sequence length for LSTM"
        )
        parser.add_argument(
            "--epochs", type=int, default=8,
            help="Epochs for LSTM/CNN"
        )
        parser.add_argument(
            "--batch", type=int, default=64,
            help="Batch size"
        )

    def handle(self, *args, **options):
        self.stdout.write(self.style.NOTICE("🚀 Starting full AI pipeline..."))

        # 1. Train ensemble models
        self.stdout.write(self.style.NOTICE("📚 Training ensemble models (RF + LSTM + CNN + Meta)..."))
        management.call_command(
            "train_ensemble",
            samples=options["samples"],
            seq_len=options["seq_len"],
            epochs=options["epochs"],
            batch=options["batch"]
        )

        # 2. Run ensemble predictions
        self.stdout.write(self.style.NOTICE("🔮 Generating predictions..."))
        management.call_command("predict")

        # 3. Refresh alerts
        self.stdout.write(self.style.NOTICE("🚨 Updating alerts..."))
        try:
            management.call_command("run_alerts")
        except Exception as e:
            self.stdout.write(self.style.WARNING(f"⚠️ Alert refresh skipped: {e}"))

        self.stdout.write(self.style.SUCCESS("✅ Pipeline complete!"))
