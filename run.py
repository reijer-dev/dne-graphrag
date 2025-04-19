from evaluation.rag_evaluater import RagEvaluater
from vanilla_rag import VanillaRag
from thuisdokterapp import MichaelRag
from lightrag_app import LightRagApp
import matplotlib.pyplot as plt
import pandas as pd

# Create evaluator
evaluater = RagEvaluater(count=100000)

# Add all RAG systems to evaluate
evaluater.add_system(MichaelRag())
# evaluater.add_system(QuickDirtyRag())
# evaluater.add_system(VoidRag())
evaluater.add_system(VanillaRag())
evaluater.add_system(LightRagApp())

# Run training evaluation (document processing)
evaluater.evaluate_training()
results_file = evaluater.evaluate_inference()

#Visualize
df = pd.read_csv('evaluation-inference.csvy')
df['memory_used'] = df['memory_used'] / 1_000_000  # Convert to MB
# Plot setup
fig, axs = plt.subplots(1, 2, figsize=(10, 4))

plt.xticks(rotation=45, ha="right")
df.set_index('rag_system')[['avg_doc_score', 'avg_global_score']].plot.bar(ax=axs[0], title="Average Answer Rating")
df.set_index('rag_system')[['avg_time']].plot.bar(ax=axs[1], color='orange', title="Average Inference Time")

axs[0].set_ylabel('rating (0-100)')
axs[0].set_xlabel('System')
axs[0].grid(True, axis='y')
axs[0].tick_params(axis='x', rotation=45)
axs[0].legend(labels=["Document questions", "Global questions"])

axs[1].set_ylabel('time (s)')
axs[1].set_xlabel('System')
axs[1].grid(True, axis='y')
axs[1].tick_params(axis='x', rotation=45)
axs[1].legend().set_visible(False)

plt.tight_layout()
#plt.show()
plt.savefig("inference.png")


# Plot setup
fig, axs = plt.subplots(1, 2, figsize=(10, 4))

plt.xticks(rotation=45, ha="right")
df.set_index('rag_system')[['memory_used']].plot.bar(ax=axs[0], title="Total Memory Used")
df.set_index('rag_system')[['training_time']].plot.bar(ax=axs[1], color='orange', title="Total Training Time")

axs[0].set_ylabel('memory (MB)')
axs[0].set_xlabel('System')
axs[0].grid(True, axis='y')
axs[0].tick_params(axis='x', rotation=45)
axs[0].legend().set_visible(False)

axs[1].set_ylabel('time (s)')
axs[1].set_xlabel('System')
axs[1].grid(True, axis='y')
axs[1].tick_params(axis='x', rotation=45)
axs[1].legend().set_visible(False)

plt.tight_layout()
#plt.show()
plt.savefig("training.png")