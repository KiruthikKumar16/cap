import os

target_tex = r"c:\Users\Kiruthik Kumar M\Downloads\Report Template_Inhouse\Thesis_content\appendix\sourcecode.tex"
project_dir = r"c:\Users\Kiruthik Kumar M\cap"

tex_header = r"""\begin{center}
	{\Large \textbf{Appendix A}}-\vspace*{0.5 cm}
	{\Large \textbf{Project Source Code}}
	
\end{center}

This appendix contains the complete source code for the MAPPO-STGNN Traffic Signal Control project, including the environment, core algorithms, baselines, and evaluation scripts.

"""

def gather_python_files(base_dir):
    files_to_include = []
    if not os.path.exists(base_dir):
        return []
    for root, _, files in os.walk(base_dir):
        # Exclude common unneeded dirs
        if "venv" in root or ".git" in root or "__pycache__" in root or "wandb" in root:
            continue
        for f in files:
            if f.endswith(".py"):
                files_to_include.append(os.path.join(root, f))
    return files_to_include

def generate():
    all_files = gather_python_files(os.path.join(project_dir, "src"))
    all_files += gather_python_files(os.path.join(project_dir, "scripts"))
    
    # Also grab any python files right in the root (if any)
    for f in os.listdir(project_dir):
        full = os.path.join(project_dir, f)
        if f.endswith('.py') and os.path.isfile(full):
            # prevent adding scripts that are just scratch
            if not f.startswith("scratch_") and f != "generate_tex_appendix.py":
                all_files.append(full)
    
    # Make list unique
    all_files = list(set(all_files))
    
    with open(target_tex, "w", encoding="utf-8") as out:
        out.write(tex_header)
        for filepath in sorted(all_files):
            rel_path = os.path.relpath(filepath, project_dir)
            # Escape underscores for LaTeX text rendering
            header_name = rel_path.replace('_', r'\_').replace('\\', '/')
            
            out.write(f"\\subsection*{{File: {header_name}}}\n")
            out.write(f"\\begin{{lstlisting}}[language=Python, caption={{{header_name}}}]\n")
            
            try:
                with open(filepath, "r", encoding="utf-8") as inf:
                    content = inf.read()
                    out.write(content)
            except Exception as e:
                out.write(f"# Error reading file {rel_path}: {str(e)}")
                  
            out.write(f"\n\\end{{lstlisting}}\n\n")

if __name__ == "__main__":
    generate()
    print("Done generating Appendix!")
