import subprocess, os

# Ruta absoluta del proyecto (backend folder)
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))

def run_subagent(name: str, task: str, recording: str):
    """Lanza un sub‑agente usando la herramienta `browser_subagent`.
    Cada sub‑agente genera un video (*.webm) que podrás revisar.
    """
    cmd = [
        "python3",
        "-m",
        "cursor.tools.browser_subagent",
        "--name", name,
        "--task", task,
        "--recording", recording,
        "--cwd", PROJECT_ROOT,
    ]
    # Ejecutamos de forma asíncrona para que no bloquee el proceso principal
    subprocess.Popen(cmd)

if __name__ == "__main__":
    # 1️⃣ UI‑Fixer → llama a qween3_wrapper.py con un prompt descriptivo
    ui_prompt = "Mejora la visualización del gráfico de velas: colores más vivos, fondo oscuro y tooltip legible."
    ui_task = f"python3 {os.path.join(PROJECT_ROOT, 'qween3_wrapper.py')} \"{ui_prompt}\""
    run_subagent("UI‑Fixer", ui_task, "ui_fix")

    # 2️⃣ Trainer → ejecuta kimi_trainer.py (entrenamiento del modelo)
    trainer_task = f"python3 {os.path.join(PROJECT_ROOT, 'kimi_trainer.py')}"
    run_subagent("Trainer", trainer_task, "trainer_run")
