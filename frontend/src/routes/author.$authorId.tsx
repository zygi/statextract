import { createFileRoute, Link } from '@tanstack/react-router'
import { getAuthorAuthorAuthorIdGet } from '../generated_types'
import { useQuery } from '@tanstack/react-query'
import { stripId as stripId } from '../utils'
import { getKeyValue, Table, TableBody, TableCell, TableColumn, TableHeader, TableRow } from '@nextui-org/react'

export const Route = createFileRoute('/author/$authorId')({
  component: AuthorComponent,
})


function AuthorComponent() {
  const { authorId } = Route.useParams()
  // // Split by / and take the last element
  // const procId = id.split('/').pop()

  // const test = getAuthorAuthorAuthorIdGet({ path: { author_id: "Asdf" } });

  const query = useQuery({
    queryKey: ['author', authorId],
    queryFn: ({ queryKey }) =>
      getAuthorAuthorAuthorIdGet({ path: { author_id: queryKey[1] } }),
  })
  if (query.isLoading) {
    return <div>Loading...</div>
  }

  const columns = [
    {
      name: 'Title',
      accessorKey: 'title',
    },
    {
      name: 'Status',
      accessorKey: 'checked',
    },
  ]
  
  return (
    <>
      <div>Name: {query.data?.data?.name}</div>
      <div>
        Works:{' '}
        <Table aria-label="Example table with dynamic content">
          <TableHeader columns={columns}>
            {(column) => (
              <TableColumn key={column.accessorKey}>
                {column.name}
              </TableColumn>
            )}
          </TableHeader>
          <TableBody items={query.data?.data?.works.filter(([work, checked]) => checked).map(([work, checked]) => ({title: work.title, checked: checked ? 'checked' : 'not checked', id: work.id}))}>
            {(dct) => (
              <TableRow key={dct.title}>
                {(columnKey) => <TableCell>
                  {columnKey === 'title' ? <Link className="underline" to={'/work/$id'} params={{"id": stripId(dct.id)}}>{dct.title}</Link> : (dct as any)[columnKey]}
                </TableCell>}
              </TableRow>
            )}
          </TableBody>
        </Table>
        {/* {query.data?.data?.works.filter(([work, checked]) => checked).map(([work, checked]) => (
          <p style={{display: checked ? 'block' : 'none'}}>
            <Link to={'/work/$id'} params={{"id": stripId(work.id)}}>{work.title}, {checked ? 'checked' : 'not checked'}</Link>
          </p>
        ))} */}
      </div>
      {/* <iframe
      src={`https://openalex.org/authors/${id}`}
        style={{ width: '100%', height: '100vh' }}
      /> */}
    </>
  )
}
