-- Migration 031: Create stored procedure for deleting projects
-- This procedure handles cascading deletion of all project-related data

CREATE OR REPLACE PROCEDURE claims.usp_DeleteProject(p_id bigint)
LANGUAGE plpgsql
AS $$
BEGIN
    -- Delete facilities associated with organizations in this project
    DELETE FROM claims.facility
    WHERE organization_id IN (
        SELECT organization_id
        FROM claims.organization
        WHERE project_id = p_id
    );

    -- Delete organizations associated with this project
    DELETE FROM claims.organization
    WHERE project_id = p_id;

    -- Delete project fields
    DELETE FROM claims.project_fields
    WHERE project_id = p_id;

    -- Delete the project itself
    DELETE FROM claims.project
    WHERE project_id = p_id;

    -- PostgreSQL automatically commits at the end of a procedure
    -- No explicit COMMIT needed in plpgsql procedures
END;
$$;
