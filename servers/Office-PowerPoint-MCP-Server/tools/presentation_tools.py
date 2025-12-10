"""
Presentation management tools for PowerPoint MCP Server.
Handles presentation creation, opening, saving, and core properties.
"""
from typing import Dict, List, Optional, Any
import os
from mcp.server.fastmcp import FastMCP
import utils as ppt_utils


def register_presentation_tools(app: FastMCP, presentations: Dict, get_current_presentation_id, get_template_search_directories):
    """Register presentation management tools with the FastMCP app"""
    # Configurable output directory for saving presentations
    OUTPUT_DIR = os.path.abspath(os.environ.get('PPT_OUTPUT_DIR', './media'))
    ENFORCE_OUTPUT_DIR = str(os.environ.get('PPT_ENFORCE_OUTPUT_DIR', 'true')).lower() in ("1", "true", "yes", "on")

    # Ensure the output directory exists
    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
    except Exception:
        # Fallback to current directory if media cannot be created
        pass

    def _resolve_save_path(requested_path: str) -> str:
        """
        Resolve the final absolute save path based on configuration.

        - Relative paths are resolved under OUTPUT_DIR
        - Absolute paths are allowed only when not enforcing OUTPUT_DIR
        - Ensures parent directory exists
        - Avoids directory traversal outside OUTPUT_DIR when enforced
        - Adds numeric suffix if the file already exists to avoid accidental overwrite
        """
        if os.path.isabs(requested_path):
            abs_path = os.path.abspath(requested_path)
            if ENFORCE_OUTPUT_DIR:
                # Ensure absolute path is within OUTPUT_DIR
                common = os.path.commonpath([OUTPUT_DIR, abs_path])
                if common != OUTPUT_DIR:
                    raise ValueError(f"Saving outside enforced output directory is not allowed: {abs_path}")
        else:
            # Always place relative paths under OUTPUT_DIR
            abs_path = os.path.abspath(os.path.join(OUTPUT_DIR, requested_path))

        # Ensure parent directory exists
        os.makedirs(os.path.dirname(abs_path), exist_ok=True)

        # If enforcement is on, double-check traversal after normalization
        if ENFORCE_OUTPUT_DIR:
            common_after = os.path.commonpath([OUTPUT_DIR, abs_path])
            if common_after != OUTPUT_DIR:
                raise ValueError(f"Resolved path escapes enforced output directory: {abs_path}")

        # Auto-avoid overwriting by appending -1, -2, ... if needed
        base, ext = os.path.splitext(abs_path)
        candidate = abs_path
        suffix = 1
        while os.path.exists(candidate):
            candidate = f"{base}-{suffix}{ext}"
            suffix += 1
        return candidate
    
    @app.tool()
    def create_presentation(id: Optional[str] = None) -> Dict:
        """
        Create a new empty PowerPoint presentation.

        Args:
            id: Optional presentation identifier; auto-generated if omitted.

        Returns:
            Dict: Presentation ID, status message, and initial slide count.
        """
        # Create a new presentation
        pres = ppt_utils.create_presentation()
        
        # Generate an ID if not provided
        if id is None:
            id = f"presentation_{len(presentations) + 1}"
        
        # Store the presentation
        presentations[id] = pres
        # Set as current presentation (this would need to be handled by caller)
        
        return {
            "presentation_id": id,
            "message": f"Created new presentation with ID: {id}",
            "slide_count": len(pres.slides)
        }

    @app.tool()
    def create_presentation_from_template(template_path: str, id: Optional[str] = None) -> Dict:
        """
        Create a new presentation from a template file.

        Args:
            template_path: Path or filename for a .pptx/.potx template.
            id: Optional presentation identifier; auto-generated if omitted.

        Returns:
            Dict: Presentation ID, resolved template path, slide count, or an error.
        """
        # Check if template file exists
        if not os.path.exists(template_path):
            # Try to find the template by searching in configured directories
            search_dirs = get_template_search_directories()
            template_name = os.path.basename(template_path)
            
            for directory in search_dirs:
                potential_path = os.path.join(directory, template_name)
                if os.path.exists(potential_path):
                    template_path = potential_path
                    break
            else:
                env_path_info = f" (PPT_TEMPLATE_PATH: {os.environ.get('PPT_TEMPLATE_PATH', 'not set')})" if os.environ.get('PPT_TEMPLATE_PATH') else ""
                return {
                    "error": f"Template file not found: {template_path}. Searched in {', '.join(search_dirs)}{env_path_info}"
                }
        
        # Create presentation from template
        try:
            pres = ppt_utils.create_presentation_from_template(template_path)
        except Exception as e:
            return {
                "error": f"Failed to create presentation from template: {str(e)}"
            }
        
        # Generate an ID if not provided
        if id is None:
            id = f"presentation_{len(presentations) + 1}"
        
        # Store the presentation
        presentations[id] = pres
        
        return {
            "presentation_id": id,
            "message": f"Created new presentation from template '{template_path}' with ID: {id}",
            "template_path": template_path,
            "slide_count": len(pres.slides),
            "layout_count": len(pres.slide_layouts)
        }

    @app.tool()
    def open_presentation(file_path: str, id: Optional[str] = None) -> Dict:
        """
        Open an existing PowerPoint presentation from disk.

        Args:
            file_path: Absolute or relative .pptx path.
            id: Optional presentation identifier; auto-generated if omitted.

        Returns:
            Dict: Presentation ID, slide count, and status or an error.
        """
        # Check if file exists
        if not os.path.exists(file_path):
            return {
                "error": f"File not found: {file_path}"
            }
        
        # Open the presentation
        try:
            pres = ppt_utils.open_presentation(file_path)
        except Exception as e:
            return {
                "error": f"Failed to open presentation: {str(e)}"
            }
        
        # Generate an ID if not provided
        if id is None:
            id = f"presentation_{len(presentations) + 1}"
        
        # Store the presentation
        presentations[id] = pres
        
        return {
            "presentation_id": id,
            "message": f"Opened presentation from {file_path} with ID: {id}",
            "slide_count": len(pres.slides)
        }

    @app.tool()
    def save_presentation(file_path: str, presentation_id: Optional[str] = None) -> Dict:
        """
        Save a presentation to a file.

        Args:
            file_path: Destination .pptx path (relative paths use configured output dir).
            presentation_id: Target presentation identifier; defaults to current.

        Returns:
            Dict: Saved path, file URI, and status or an error.
        """
        # Use the specified presentation or the current one
        pres_id = presentation_id if presentation_id is not None else get_current_presentation_id()
        
        if pres_id is None or pres_id not in presentations:
            return {
                "error": "No presentation is currently loaded or the specified ID is invalid"
            }
        
        # Resolve final path and save the presentation
        try:
            resolved_path = _resolve_save_path(file_path)
            saved_path = ppt_utils.save_presentation(presentations[pres_id], resolved_path)
            # Build URI for easier downstream reference
            uri = f"file://{saved_path}"
            return {
                "message": f"Presentation saved to {saved_path}",
                "file_path": saved_path,
                "abs_path": saved_path,
                "uri": uri
            }
        except Exception as e:
            return {
                "error": f"Failed to save presentation: {str(e)}"
            }

    @app.tool()
    def get_presentation_info(presentation_id: Optional[str] = None) -> Dict:
        """
        Get basic metadata and layout information for a presentation.

        Args:
            presentation_id: Target presentation identifier; defaults to current.

        Returns:
            Dict: Presentation ID, counts, and layout details or an error.
        """
        pres_id = presentation_id if presentation_id is not None else get_current_presentation_id()
        
        if pres_id is None or pres_id not in presentations:
            return {
                "error": "No presentation is currently loaded or the specified ID is invalid"
            }
        
        pres = presentations[pres_id]
        
        try:
            info = ppt_utils.get_presentation_info(pres)
            info["presentation_id"] = pres_id
            return info
        except Exception as e:
            return {
                "error": f"Failed to get presentation info: {str(e)}"
            }

    @app.tool()
    def get_template_file_info(template_path: str) -> Dict:
        """
        Get layout and property information for a template file.

        Args:
            template_path: Template path or filename; searched in configured directories if needed.

        Returns:
            Dict: Template layouts, counts, and properties or an error.
        """
        # Check if template file exists
        if not os.path.exists(template_path):
            # Try to find the template by searching in configured directories
            search_dirs = get_template_search_directories()
            template_name = os.path.basename(template_path)
            
            for directory in search_dirs:
                potential_path = os.path.join(directory, template_name)
                if os.path.exists(potential_path):
                    template_path = potential_path
                    break
            else:
                return {
                    "error": f"Template file not found: {template_path}. Searched in {', '.join(search_dirs)}"
                }
        
        try:
            return ppt_utils.get_template_info(template_path)
        except Exception as e:
            return {
                "error": f"Failed to get template info: {str(e)}"
            }

    @app.tool()
    def set_core_properties(
        title: Optional[str] = None,
        subject: Optional[str] = None,
        author: Optional[str] = None,
        keywords: Optional[str] = None,
        comments: Optional[str] = None,
        presentation_id: Optional[str] = None
    ) -> Dict:
        """
        Set core document properties on a presentation.

        Args:
            title: Document title.
            subject: Document subject.
            author: Author name.
            keywords: Comma-separated keywords.
            comments: Free-form comments/notes.
            presentation_id: Target presentation identifier; defaults to current.

        Returns:
            Dict: Status message or an error.
        """
        pres_id = presentation_id if presentation_id is not None else get_current_presentation_id()
        
        if pres_id is None or pres_id not in presentations:
            return {
                "error": "No presentation is currently loaded or the specified ID is invalid"
            }
        
        pres = presentations[pres_id]
        
        try:
            ppt_utils.set_core_properties(
                pres,
                title=title,
                subject=subject,
                author=author,
                keywords=keywords,
                comments=comments
            )
            
            return {
                "message": "Core properties updated successfully"
            }
        except Exception as e:
            return {
                "error": f"Failed to set core properties: {str(e)}"
            }