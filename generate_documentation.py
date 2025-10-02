import re
import sys
import subprocess
import os
import webbrowser
from pathlib import Path

try:
    import markdown
except ImportError:
    print("The 'markdown' package is required. Install it via 'pip install markdown'.", file=sys.stderr)
    sys.exit(1)

# Complete module structure based on your file tree
DOCUMENTATION_STRUCTURE = {
    "Overview": {
        "type": "single",
        "link": "index.html",
        "icon": "🏠"
    },
    "Port and channel": {
        "type": "section", 
        "icon": "🚢",
        "modules": [
            ("Port", "simulation_classes/port.html"),
            ("Channel", "simulation_classes/channel.html"),
        ]
    },
    "Terminal": {
        "type": "section",
        "icon": "🏗",
        "modules": [
            ("Container", "simulation_classes/terminal_container.html"),
            ("Dry bulk", "simulation_classes/terminal_drybulk.html"),
            ("Liquid", "simulation_classes/terminal_liquid.html")
        ]
    },
    "Landside": {
        "type": "section",
        "icon": "🚛",
        "modules": [
            ("Truck", "simulation_classes/truck.html"),
            ("Train", "simulation_classes/train.html"),
            ("Pipeline", "simulation_classes/pipeline.html")
        ]
    },
    "Simulation helpers": {
        "type": "section",
        "icon": "⚙️",
        "modules": [
            ("Run", "simulation_handler/run_simulation.html"),
            ("Generators", "simulation_handler/generators.html"),
            ("Helpers", "simulation_handler/helpers.html"),
            ("Preprocess", "simulation_handler/preprocess.html")
        ]
    },
    "Analysis": {
        "type": "section",
        "icon": "📊",
        "modules": [
            ("Capacity", "simulation_analysis/capacity.html"),
            ("Utilization", "simulation_analysis/resource_utilization.html"),
            ("Results", "simulation_analysis/results.html"),
            ("Collate", "simulation_analysis/collate_results.html"),
            ("What-If", "simulation_analysis/whatif_scenarios.html")
        ]
    },
}

# List of all modules for pdoc
MODULES = [
    "config",
    "constants", 
    "inputs",
    "main",
    "simulation_classes",
    "simulation_handler",
    "simulation_analysis"
]

def get_venv_python() -> str:
    """Return the path to the current virtualenv's python executable."""
    exe = Path(sys.executable)
    if "Scripts" in exe.parts or "bin" in exe.parts:
        return str(exe)
    venv_dir = Path(__file__).parent / ".venv"
    if os.name == "nt":
        return str(venv_dir / "Scripts" / "python.exe")
    else:
        return str(venv_dir / "bin" / "python")

def generate_docs(output_dir: str = "./simulation_documentation"):
    """Run pdoc to generate documentation for all modules and packages."""
    python_exe = get_venv_python()
    
    cmd = [
        python_exe, "-m", "pdoc",
        *MODULES,
        "--docformat", "google",
        "--no-include-undocumented",
        "-o", output_dir,
        "--logo", "spartaStacked.png"
    ]
    
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).parent)
    
    print(f"Running pdoc...")
    subprocess.run(cmd, check=True, env=env)
    print(f"Documentation generated into {output_dir}")

def replace_sidebar_in_html(output_dir: str = "./simulation_documentation"):
    """Replace the pdoc sidebar with a custom one in all HTML files."""
    
    output_path = Path(output_dir)
    
    # Process all HTML files
    for html_file in output_path.rglob("*.html"):
        try:
            content = html_file.read_text(encoding="utf-8")
            
            # Calculate relative path for links
            depth = len(html_file.relative_to(output_path).parts) - 1
            prefix = "../" * depth if depth > 0 else ""
            
            # Build custom navigation HTML with hamburger menu for mobile
            nav_html = f"""
<!-- Mobile Menu Toggle -->
<div id="mobile-menu-toggle" style="
    display: none;
    position: fixed;
    top: 10px;
    right: 10px;
    z-index: 1001;
    width: 40px;
    height: 40px;
    background: #2c3e50;
    border-radius: 5px;
    cursor: pointer;
    box-shadow: 0 2px 5px rgba(0,0,0,0.2);
">
    <div style="
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
    ">
        <div class="hamburger-line" style="
            width: 24px;
            height: 3px;
            background: white;
            margin: 5px 0;
            transition: 0.3s;
        "></div>
        <div class="hamburger-line" style="
            width: 24px;
            height: 3px;
            background: white;
            margin: 5px 0;
            transition: 0.3s;
        "></div>
        <div class="hamburger-line" style="
            width: 24px;
            height: 3px;
            background: white;
            margin: 5px 0;
            transition: 0.3s;
        "></div>
    </div>
</div>

<!-- Overlay for mobile menu -->
<div id="mobile-overlay" style="
    display: none;
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: rgba(0,0,0,0.5);
    z-index: 999;
"></div>

<!-- Custom Navigation Start -->
<nav id="custom-sidebar" style="
    position: fixed;
    left: 0;
    top: 0;
    width: 260px;
    height: 100vh;
    background: linear-gradient(180deg, #2c3e50 0%, #34495e 100%);
    color: white;
    overflow-y: auto;
    box-shadow: 2px 0 10px rgba(0,0,0,0.2);
    z-index: 1000;
    transition: transform 0.3s ease;
">
    <!-- Close button for mobile -->
    <div id="mobile-close" style="
        display: none;
        position: absolute;
        top: 10px;
        right: 10px;
        width: 30px;
        height: 30px;
        cursor: pointer;
        color: white;
        font-size: 24px;
        line-height: 30px;
        text-align: center;
    ">×</div>
    
    <div style="padding: 1.5rem; background: rgba(0,0,0,0.2); border-bottom: 2px solid #3498db;">
        <h2 style="margin: 0; font-size: 1.4rem; display: flex; align-items: center; gap: 0.5rem;">
            Modules
        </h2>
    </div>
    <div style="padding: 0.5rem 0;">
"""
            
            for section_name, section_data in DOCUMENTATION_STRUCTURE.items():
                icon = section_data.get("icon", "📄")
                
                if section_data["type"] == "single":
                    link = prefix + section_data["link"]
                    # Check if this is the active page
                    is_active = section_data["link"] in str(html_file.name)
                    active_style = "background: rgba(52, 152, 219, 0.3); border-left: 4px solid #3498db;" if is_active else ""
                    
                    nav_html += f"""
        <div style="border-bottom: 1px solid rgba(255,255,255,0.1);">
            <a href="{link}" class="nav-link" style="
                display: flex;
                align-items: center;
                gap: 0.5rem;
                padding: 0.8rem 1rem;
                color: white;
                text-decoration: none;
                transition: background 0.2s;
                {active_style}
            " onmouseover="this.style.background='rgba(52,152,219,0.2)'" 
               onmouseout="this.style.background='{active_style if is_active else ''}'">
                <span>{icon}</span>
                <span>{section_name}</span>
            </a>
        </div>
"""
                else:
                    # Section with multiple items
                    section_id = section_name.replace(" ", "_")
                    nav_html += f"""
        <div style="border-bottom: 1px solid rgba(255,255,255,0.1);">
            <div onclick="toggleSection('{section_id}')" style="
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding: 0.8rem 1rem;
                cursor: pointer;
                background: rgba(0,0,0,0.1);
                transition: background 0.2s;
            " onmouseover="this.style.background='rgba(52,152,219,0.2)'" 
               onmouseout="this.style.background='rgba(0,0,0,0.1)'">
                <span style="display: flex; align-items: center; gap: 0.5rem;">
                    <span>{icon}</span>
                    <span>{section_name}</span>
                </span>
                <span id="arrow_{section_id}" style="transition: transform 0.2s;">▶</span>
            </div>
            <div id="{section_id}" style="display: none; background: rgba(0,0,0,0.2);">
"""
                    
                    for item_name, item_link in section_data["modules"]:
                        link = prefix + item_link
                        # Check if this is the active page
                        is_active = item_link.split('/')[-1] in str(html_file.name)
                        active_style = "background: rgba(52, 152, 219, 0.4); font-weight: bold;" if is_active else ""
                        
                        nav_html += f"""
                <a href="{link}" class="nav-link" style="
                    display: block;
                    padding: 0.6rem 1rem 0.6rem 2.5rem;
                    color: rgba(255,255,255,0.9);
                    text-decoration: none;
                    transition: all 0.2s;
                    {active_style}
                " onmouseover="this.style.background='rgba(52,152,219,0.3)'; this.style.paddingLeft='2.7rem'" 
                   onmouseout="this.style.background='{active_style if is_active else ''}'; this.style.paddingLeft='2.5rem'">
                    {item_name}
                </a>
"""
                    
                    nav_html += """
            </div>
        </div>
"""
            
            nav_html += """
    </div>
</nav>
<!-- Custom Navigation End -->

<script>
function toggleSection(sectionId) {
    var section = document.getElementById(sectionId);
    var arrow = document.getElementById('arrow_' + sectionId);
    if (section.style.display === 'none' || section.style.display === '') {
        section.style.display = 'block';
        arrow.style.transform = 'rotate(90deg)';
    } else {
        section.style.display = 'none';
        arrow.style.transform = 'rotate(0deg)';
    }
}

// Mobile menu functionality
function initMobileMenu() {
    var sidebar = document.getElementById('custom-sidebar');
    var menuToggle = document.getElementById('mobile-menu-toggle');
    var overlay = document.getElementById('mobile-overlay');
    var closeBtn = document.getElementById('mobile-close');
    
    // Toggle menu
    if (menuToggle) {
        menuToggle.addEventListener('click', function() {
            sidebar.style.transform = 'translateX(0)';
            overlay.style.display = 'block';
        });
    }
    
    // Close menu functions
    function closeMenu() {
        sidebar.style.transform = 'translateX(-100%)';
        overlay.style.display = 'none';
    }
    
    if (overlay) {
        overlay.addEventListener('click', closeMenu);
    }
    
    if (closeBtn) {
        closeBtn.addEventListener('click', closeMenu);
    }
    
    // Close menu when clicking on a link (mobile only)
    if (window.innerWidth <= 768) {
        var navLinks = document.querySelectorAll('.nav-link');
        navLinks.forEach(function(link) {
            link.addEventListener('click', function() {
                setTimeout(closeMenu, 100);
            });
        });
    }
}

// Auto-expand active section
document.addEventListener('DOMContentLoaded', function() {
    // Hide original pdoc navigation
    var originalNav = document.querySelector('body > nav');
    if (originalNav) {
        originalNav.style.display = 'none';
    }
    
    // Adjust main content margin
    var mainContent = document.querySelector('main');
    if (mainContent) {
        mainContent.style.marginLeft = '260px';
        mainContent.style.padding = '2rem';
    }
    
    // Auto-expand sections with active items
    var activeLinks = document.querySelectorAll('a[style*="background: rgba(52, 152, 219, 0.4)"]');
    activeLinks.forEach(function(link) {
        var parentSection = link.parentElement;
        if (parentSection && parentSection.id) {
            parentSection.style.display = 'block';
            var arrow = document.getElementById('arrow_' + parentSection.id);
            if (arrow) {
                arrow.style.transform = 'rotate(90deg)';
            }
        }
    });
    
    // Initialize mobile menu
    initMobileMenu();
    
    // Check if mobile on load
    checkMobile();
});

// Check if mobile and adjust styles
function checkMobile() {
    var sidebar = document.getElementById('custom-sidebar');
    var menuToggle = document.getElementById('mobile-menu-toggle');
    var closeBtn = document.getElementById('mobile-close');
    var mainContent = document.querySelector('main');
    
    if (window.innerWidth <= 768) {
        // Mobile view
        if (sidebar) sidebar.style.transform = 'translateX(-100%)';
        if (menuToggle) menuToggle.style.display = 'block';
        if (closeBtn) closeBtn.style.display = 'block';
        if (mainContent) {
            mainContent.style.marginLeft = '0';
            mainContent.style.padding = '60px 1rem 1rem 1rem';
        }
    } else {
        // Desktop view
        if (sidebar) sidebar.style.transform = 'translateX(0)';
        if (menuToggle) menuToggle.style.display = 'none';
        if (closeBtn) closeBtn.style.display = 'none';
        if (mainContent) {
            mainContent.style.marginLeft = '260px';
            mainContent.style.padding = '2rem';
        }
    }
}

// Listen for window resize
window.addEventListener('resize', checkMobile);
</script>

<style>
/* Additional styles to ensure visibility */
body {
    margin: 0;
    padding: 0;
}

/* Hide pdoc nav */
body > nav.pdoc,
nav.pdoc {
    display: none !important;
}

/* Ensure main content has proper margin and text wrapping */
main.pdoc,
main {
    margin-left: 260px !important;
    padding: 2rem !important;
    max-width: 100%;
    overflow-x: hidden;
}

/* Fix text wrapping for all content */
main * {
    word-wrap: break-word;
    word-break: break-word;
    overflow-wrap: break-word;
}

/* Fix code blocks to wrap properly */
pre {
    white-space: pre-wrap !important;
    word-wrap: break-word !important;
    overflow-x: auto !important;
    max-width: 100% !important;
}

code {
    word-wrap: break-word !important;
    overflow-wrap: break-word !important;
}

/* Fix tables to be responsive */
table {
    display: block;
    overflow-x: auto;
    max-width: 100%;
}

/* Ensure custom sidebar is visible */
#custom-sidebar {
    display: block !important;
    visibility: visible !important;
}

/* Mobile responsive styles */
@media (max-width: 768px) {
    #custom-sidebar {
        transform: translateX(-100%);
        transition: transform 0.3s ease;
    }
    
    #mobile-menu-toggle {
        display: block !important;
    }
    
    #mobile-close {
        display: block !important;
    }
    
    main.pdoc,
    main {
        margin-left: 0 !important;
        padding: 60px 1rem 1rem 1rem !important;
    }
    
    /* Ensure content fits on mobile */
    main {
        width: 100%;
        box-sizing: border-box;
    }
    
    /* Fix wide elements on mobile */
    img, video, iframe {
        max-width: 100%;
        height: auto;
    }
    
    /* Adjust font size for better mobile reading */
    body {
        font-size: 14px;
    }
    
    /* Fix code blocks on mobile */
    pre {
        padding: 0.5rem;
        font-size: 12px;
    }
}

/* Tablet responsive */
@media (min-width: 769px) and (max-width: 1024px) {
    main.pdoc,
    main {
        margin-left: 260px !important;
        padding: 1.5rem !important;
    }
}

/* Animation for hamburger menu */
#mobile-menu-toggle:active .hamburger-line {
    background: #3498db;
}
</style>
"""
            
            # Insert the custom navigation right after <body> tag
            if "<body>" in content:
                # Find body tag and insert navigation after it
                body_index = content.index("<body>")
                body_end = content.index(">", body_index) + 1
                content = content[:body_end] + nav_html + content[body_end:]
            elif "<body " in content:
                # Handle body with attributes
                body_index = content.index("<body ")
                body_end = content.index(">", body_index) + 1
                content = content[:body_end] + nav_html + content[body_end:]
            
            html_file.write_text(content, encoding="utf-8")
            print(f"  ✓ Updated: {html_file.relative_to(output_path)}")
            
        except Exception as e:
            print(f"  ✗ Error processing {html_file}: {e}")

def inject_readme(
    html_path: str = "./simulation_documentation/index.html",
    readme_path: str = "README.md",
    repo_url: str = "https://github.com/spartalab/port-simulation",
    manual_url: str = "Simulation_Manual.pdf"
):
    """Inject README content into the index.html"""
    html_path = Path(html_path)
    readme_path = Path(readme_path)
    
    if not html_path.exists():
        print(f"  ✗ Index file not found")
        return
        
    if not readme_path.exists():
        print(f"  ✗ README file not found")
        return
    
    html = html_path.read_text(encoding="utf-8")
    md_text = readme_path.read_text(encoding="utf-8")
    md_html = markdown.markdown(md_text, extensions=["fenced_code", "tables"])

    # FIX: Correct the image path for the logo from the README
    md_html = md_html.replace('src="simulation_documentation/', 'src="')
    
    # Update title
    html = re.sub(r"<title>.*?</title>", "<title>Documentation</title>", html)
    
    # Add welcome section and CTAs with responsive design
    welcome_html = f"""
<div style="margin-bottom: 2rem; display: flex; gap: 1rem; flex-wrap: wrap;">

    <a href="{repo_url}" target="_blank" class="action-button primary">
        View codebase on GitHub
    </a>

    <a href="{manual_url}" target="_blank" class="action-button secondary">
        Download configuration manual (pdf)
    </a>
</div>

<style>
    .action-button {{
        display: inline-block;
        padding: 0.75rem 1.5rem;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
        font-size: 1rem;
        font-weight: 600;
        text-align: center;
        text-decoration: none;
        border-radius: 8px;
        border: 2px solid #667eea;
        transition: all 0.2s ease-in-out;
        white-space: nowrap;
    }}
    .action-button.primary {{
        background-color: #667eea;
        color: white;
    }}
    .action-button.secondary {{
        background-color: transparent;
        color: #667eea;
    }}
    .action-button.primary:hover {{
        background-color: transparent;
        border-color: #667eea;
    }}
    .action-button.secondary:hover {{
        background-color: #667eea;
        color: white;
    }}
    
    /* Mobile responsive buttons */
    @media (max-width: 768px) {{
        .action-button {{
            width: 100%;
            margin-bottom: 0.5rem;
        }}
    }}
</style>
"""
    
    # Insert content into main
    if "<main" in html and "</main>" in html:
        pattern = re.compile(r'(<main[^>]*>)(.*?)(</main>)', re.DOTALL)
        replacement = f'\\1{welcome_html}<div class="readme-content">{md_html}</div>\\2\\3'
        html = pattern.sub(replacement, html, count=1)
    
    html_path.write_text(html, encoding="utf-8")
    print(f"  ✓ README injected into index.html")

def open_index(html_path: str = "./simulation_documentation/index.html"):
    """Open the documentation in browser."""
    abs_path = Path(html_path).resolve()
    if abs_path.exists():
        webbrowser.open(f"file://{abs_path}")
        print(f"  ✓ Opening documentation in browser")
    else:
        print(f"  ✗ Could not find {abs_path}")

if __name__ == "__main__":
    print("=" * 60)
    print("🚀 PORT SIMULATION DOCUMENTATION GENERATOR")
    print("=" * 60)
    
    print("\n📚 Step 1: Generating base documentation with pdoc...")
    generate_docs()
    
    print("\n🎨 Step 2: Creating custom sidebar...")
    replace_sidebar_in_html()
    
    print("\n📝 Step 3: Adding README content...")
    inject_readme()
    
    print("\n" + "=" * 60)
    print("✅ Documentation generation complete!")
    print("=" * 60)
    
    open_index()