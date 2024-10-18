import { Divider } from '@nextui-org/react'
import { createRootRoute, Link, Outlet } from '@tanstack/react-router'
import { TanStackRouterDevtools } from '@tanstack/router-devtools'

export const Route = createRootRoute({
  component: () => (
    <>
      <div className="p-2 flex gap-2">
        <Link to="/authors" className="[&.active]:font-bold">
          Authors
        </Link>{' '}
        {/* <Link to="/author/$authorId" params={{authorId: "a5000387389"}} className="[&.active]:font-bold">
          Author 1
        </Link> */}
      </div>

      <Divider/>

      <Outlet />
      <footer className="pb-1 text-center text-default-500"></footer>
      <TanStackRouterDevtools />
    </>
  ),
})
